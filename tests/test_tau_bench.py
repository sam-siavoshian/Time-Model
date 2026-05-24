"""Unit tests for the tau_sessions external benchmark.

Covers:
  - dataset generation determinism (seed -> byte-identical JSONL)
  - dataset shape (counts per bucket + task)
  - scoring math (exact_match yes/no, log_mae, Pearson, bootstrap CI)
  - adapter loading (registry shape, missing-checkpoint error, vanilla
    + prompt instantiation contract)

Run with:
  uv run python -m pytest tests/test_tau_bench.py -v

Tests that touch the actual HF model weights are guarded by the env
flag TAU_BENCH_SMOKE_HF=1 so CI can skip them by default.

License: MIT.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import sys
import tempfile
import unittest
from collections import Counter

# Make sure we import from the repo root rather than any installed copy.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval.external import generate_tau_sessions as gen


class TestDatasetDeterminism(unittest.TestCase):
    """The whole point of seeding is that two runs produce identical
    bytes. Anything weaker than that means downstream papers can't
    reproduce numbers exactly."""

    def test_seed_42_produces_300_sessions(self):
        sessions = gen.generate(seed=42)
        self.assertEqual(len(sessions), 300)

    def test_per_bucket_50(self):
        sessions = gen.generate(seed=42)
        c = Counter(s.tau_bucket for s in sessions)
        self.assertEqual(set(c.values()), {50})
        self.assertEqual(set(c.keys()),
                         {"1s", "60s", "600s", "6h", "24h", "7d"})

    def test_per_task_100(self):
        sessions = gen.generate(seed=42)
        c = Counter(s.task_type for s in sessions)
        self.assertEqual(c["duration_recall"], 100)
        self.assertEqual(c["staleness"], 100)
        self.assertEqual(c["adaptive"], 100)

    def test_seed_reproduces_identical_jsonl(self):
        s1 = gen.generate(seed=42)
        s2 = gen.generate(seed=42)
        b1 = "\n".join(s.to_json() for s in s1)
        b2 = "\n".join(s.to_json() for s in s2)
        self.assertEqual(hashlib.sha256(b1.encode()).hexdigest(),
                         hashlib.sha256(b2.encode()).hexdigest())

    def test_different_seeds_differ(self):
        s1 = gen.generate(seed=42)
        s2 = gen.generate(seed=43)
        # Same shape, different tau values across the board.
        diffs = sum(1 for a, b in zip(s1, s2) if a.tau_seconds != b.tau_seconds)
        self.assertGreater(diffs, 290)

    def test_tau_falls_inside_bucket(self):
        sessions = gen.generate(seed=42)
        bounds = {name: (lo, hi) for name, lo, hi in gen.BUCKETS}
        for s in sessions:
            lo, hi = bounds[s.tau_bucket]
            self.assertGreaterEqual(s.tau_seconds, lo, msg=s.session_id)
            self.assertLess(s.tau_seconds, hi, msg=s.session_id)

    def test_session_ids_unique_and_sequential(self):
        sessions = gen.generate(seed=42)
        ids = [s.session_id for s in sessions]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(ids, [f"s_{i:04d}" for i in range(300)])

    def test_staleness_ground_truth_consistency(self):
        sessions = gen.generate(seed=42)
        for s in sessions:
            if s.task_type != "staleness":
                continue
            dur = float(s.extra["duration_seconds"])
            expected = "yes" if s.tau_seconds < dur else "no"
            self.assertEqual(s.ground_truth, expected, msg=s.session_id)


class TestHumanizers(unittest.TestCase):
    def test_humanize_tau_bucketing(self):
        self.assertIn("seconds", gen.humanize_tau(2.0))
        self.assertIn("seconds", gen.humanize_tau(45))
        self.assertIn("minutes", gen.humanize_tau(600))
        self.assertIn("hours", gen.humanize_tau(3600 * 5))
        self.assertIn("days", gen.humanize_tau(86400 * 3))

    def test_short_tau_compact_form(self):
        self.assertEqual(gen.short_tau(5), "5s")
        self.assertEqual(gen.short_tau(120), "2m")
        self.assertEqual(gen.short_tau(3600 * 3), "3h")
        self.assertEqual(gen.short_tau(86400 * 2), "2d")


class TestScoring(unittest.TestCase):
    """Scoring math is the only piece a third party will scrutinize when
    they reproduce numbers from this benchmark, so test every codepath."""

    def setUp(self):
        from eval.external import eval_tau_bench as e
        self.e = e

    def test_exact_match_yes_no(self):
        score = self.e.score_exact_match
        self.assertEqual(score("Yes, it is.", "yes"), 1)
        self.assertEqual(score("no.", "no"), 1)
        self.assertEqual(score("No.", "yes"), 0)
        self.assertEqual(score("I don't know", "yes"), 0)

    def test_exact_match_word_boundary(self):
        # "yesterday" should NOT match "yes".
        score = self.e.score_exact_match
        self.assertEqual(score("Yesterday it ran", "yes"), 0)
        self.assertEqual(score("Nothing changed", "no"), 0)

    def test_log_mae_perfect(self):
        err = self.e.score_log_mae("about 60 minutes", 3600.0)
        self.assertAlmostEqual(err, 0.0, places=4)

    def test_log_mae_factor_of_two(self):
        err = self.e.score_log_mae("about 30 minutes", 3600.0)
        # log10(1800) - log10(3600) = -log10(2) ~= -0.301
        self.assertAlmostEqual(err, math.log10(2.0), places=4)

    def test_log_mae_parse_fail_returns_nan(self):
        err = self.e.score_log_mae("uh I dunno", 3600.0)
        self.assertTrue(err != err)                          # NaN

    def test_pearson_perfect(self):
        r = self.e.pearson([1, 2, 3, 4], [10, 20, 30, 40])
        self.assertAlmostEqual(r, 1.0, places=4)

    def test_pearson_perfect_negative(self):
        r = self.e.pearson([1, 2, 3, 4], [4, 3, 2, 1])
        self.assertAlmostEqual(r, -1.0, places=4)

    def test_pearson_zero_variance(self):
        r = self.e.pearson([1, 1, 1, 1], [1, 2, 3, 4])
        self.assertEqual(r, 0.0)

    def test_pearson_too_short(self):
        self.assertEqual(self.e.pearson([1], [2]), 0.0)
        self.assertEqual(self.e.pearson([], []), 0.0)

    def test_bootstrap_ci_basic(self):
        import statistics
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        lo, hi = self.e.bootstrap_ci(values, statistics.mean,
                                     n_boot=500, seed=42)
        # The true mean is 3.0; a 95% bootstrap CI should bracket it.
        self.assertLess(lo, 3.0)
        self.assertGreater(hi, 3.0)

    def test_bootstrap_ci_empty(self):
        import statistics
        lo, hi = self.e.bootstrap_ci([], statistics.mean,
                                     n_boot=100, seed=42)
        self.assertTrue(lo != lo and hi != hi)

    def test_bootstrap_ci_deterministic_with_seed(self):
        import statistics
        v = [0.1, 0.2, 0.5, 0.7, 0.9]
        ci_a = self.e.bootstrap_ci(v, statistics.mean, n_boot=300, seed=42)
        ci_b = self.e.bootstrap_ci(v, statistics.mean, n_boot=300, seed=42)
        self.assertEqual(ci_a, ci_b)

    def test_bootstrap_pairs_ci_pearson(self):
        pairs = [(i, 2 * i + 0.01 * (i % 3)) for i in range(30)]
        lo, hi = self.e.bootstrap_pairs_ci(pairs, self.e.pearson,
                                           n_boot=300, seed=42)
        # Strongly positive correlation -> CI well above 0.
        self.assertGreater(lo, 0.5)
        self.assertLessEqual(hi, 1.0 + 1e-9)


class TestAggregate(unittest.TestCase):
    """Aggregation is the bridge between raw rows and the headline
    numbers. A bug here corrupts every report."""

    def setUp(self):
        from eval.external import eval_tau_bench as e
        self.e = e

    def _rows(self, n_dr=10, n_st=10, n_ad=10):
        rows = []
        # duration_recall: tiny log error
        for i in range(n_dr):
            rows.append({
                "session_id": f"dr_{i}", "tau_bucket": "60s",
                "tau_seconds": 60.0, "task_type": "duration_recall",
                "eval_protocol": "mae",
                "prompt": "p", "ground_truth": "gt", "prediction": "60s",
                "log_abs_err": 0.1, "parsed_seconds": 60.0,
            })
        # staleness: 80% accurate
        for i in range(n_st):
            rows.append({
                "session_id": f"st_{i}", "tau_bucket": "60s",
                "tau_seconds": 60.0, "task_type": "staleness",
                "eval_protocol": "exact_match",
                "prompt": "p", "ground_truth": "yes", "prediction": "yes",
                "score": 1 if i < int(0.8 * n_st) else 0,
            })
        # adaptive: log-tau vs log-len perfectly correlated
        for i in range(n_ad):
            rows.append({
                "session_id": f"ad_{i}", "tau_bucket": "60s",
                "tau_seconds": 10 ** (i + 1), "task_type": "adaptive",
                "eval_protocol": "len_elasticity",
                "prompt": "p", "ground_truth": "", "prediction": "x" * (i + 1),
                "response_length": 10 ** (i + 1),
            })
        return rows

    def test_aggregate_shape(self):
        agg = self.e.aggregate(self._rows(), n_boot=200)
        self.assertEqual(agg["n_sessions"], 30)
        self.assertIn("staleness", agg["per_task"])
        self.assertIn("duration_recall", agg["per_task"])
        self.assertIn("adaptive", agg["per_task"])

    def test_aggregate_staleness_acc(self):
        agg = self.e.aggregate(self._rows(n_st=10), n_boot=200)
        self.assertAlmostEqual(agg["per_task"]["staleness"]["accuracy"],
                               0.8, places=4)

    def test_aggregate_duration_mae(self):
        agg = self.e.aggregate(self._rows(n_dr=5), n_boot=200)
        self.assertAlmostEqual(agg["per_task"]["duration_recall"]["log_mae"],
                               0.1, places=4)

    def test_aggregate_adaptive_pearson_positive(self):
        agg = self.e.aggregate(self._rows(n_ad=10), n_boot=200)
        # log10(tau) and log10(length) are perfectly correlated by construction.
        self.assertGreater(agg["per_task"]["adaptive"]["pearson_r"], 0.99)

    def test_composite_score_in_unit(self):
        agg = self.e.aggregate(self._rows(), n_boot=200)
        self.assertGreaterEqual(agg["composite_score"], 0.0)
        self.assertLessEqual(agg["composite_score"], 1.0)


class TestAdapterRegistry(unittest.TestCase):
    """The adapter registry is the contract surface for third-party
    contributors. Test that it advertises the right names and that the
    CI adapter refuses to load without a checkpoint."""

    def test_registry_has_three_adapters(self):
        from eval.external.adapters import ADAPTERS
        self.assertEqual(set(ADAPTERS), {"vanilla", "prompt", "ci"})

    def test_unknown_adapter_raises(self):
        from eval.external.adapters import load_adapter
        with self.assertRaises(KeyError):
            load_adapter("nonexistent")

    def test_ci_adapter_requires_checkpoint(self):
        from eval.external.adapters.ci_adapter import CIAdapter
        with self.assertRaises(ValueError):
            CIAdapter(base_model="Qwen/Qwen2.5-3B-Instruct")

    def test_prompt_adapter_format_elapsed(self):
        from eval.external.adapters.prompt_adapter import _format_elapsed
        self.assertEqual(_format_elapsed(5.0), "5s")
        self.assertEqual(_format_elapsed(125.0), "2m 5s")
        self.assertIn("h", _format_elapsed(3700.0))
        self.assertIn("d", _format_elapsed(86400 * 2 + 3600))


class TestRoundTripJsonl(unittest.TestCase):
    """Writing and re-reading the JSONL must round-trip every field."""

    def test_write_then_load(self):
        from eval.external.eval_tau_bench import load_sessions
        sessions = gen.generate(seed=42)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.jsonl")
            gen.write_jsonl(sessions, path)
            loaded = load_sessions(path)
        self.assertEqual(len(loaded), 300)
        # Spot-check round-trip on first session.
        self.assertEqual(loaded[0]["session_id"], sessions[0].session_id)
        self.assertEqual(loaded[0]["task_type"], sessions[0].task_type)
        self.assertEqual(loaded[0]["tau_bucket"], sessions[0].tau_bucket)


@unittest.skipUnless(os.environ.get("TAU_BENCH_SMOKE_HF") == "1",
                     "set TAU_BENCH_SMOKE_HF=1 to run HF-touching tests")
class TestVanillaSmoke(unittest.TestCase):
    """One-session end-to-end smoke for the vanilla adapter. This
    actually downloads + loads the base model so it's gated behind an
    env flag; the regular test suite never hits the network."""

    def test_vanilla_generates_string(self):
        from eval.external.adapters import load_adapter
        ad = load_adapter("vanilla",
                          base_model=os.environ.get("TAU_BENCH_BASE",
                                                    "Qwen/Qwen2.5-3B-Instruct"),
                          device="cpu",
                          dtype="fp32",
                          max_new_tokens=8)
        try:
            out = ad.generate("Say hi.", tau_seconds=12.0)
            self.assertIsInstance(out, str)
        finally:
            ad.cleanup()


if __name__ == "__main__":
    unittest.main()
