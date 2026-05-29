from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from eval.tps.analyze import monotonicity
from eval.tps.benchmark import FAMILIES, TAU_VALUES_S, validate_thresholds
from eval.tps.training_data import build_records
from model.qwen_time_data import (
    gen_clock_conversation,
    gen_phase_conversation,
    gen_silent_gap_conversation,
)


ROOT = Path(__file__).resolve().parents[1]


def test_track_a_generators_are_only_clock_silent_gap_phase():
    import random

    rng = random.Random(0)
    modes = {
        gen_clock_conversation(rng)["mode"],
        gen_silent_gap_conversation(rng)["mode"],
        gen_phase_conversation(rng)["mode"],
    }
    assert modes == {"clock", "silent_gap", "phase"}


def test_track_b_training_records_are_single_letter_tps_policy_only():
    records = build_records(split="train", seed=0)
    assert records
    assert {rec["track"] for rec in records} == {"track_b_policy"}
    assert {rec["mode"] for rec in records} == {"tps_policy_hidden_only"}
    for rec in records:
        assert re.fullmatch(r"[ABCD]<\|im_end\|>", rec["answer_text"])
        assert rec["answer"] in {"A", "B", "C", "D"}
        assert rec["mode"] not in {"clock", "silent_gap", "phase"}


def test_track_b_train_split_excludes_heldout_templates_and_family():
    records = build_records(split="train", seed=11)
    assert records
    assert all(not rec["held_out_template"] for rec in records)
    assert all(not rec["held_out_family"] for rec in records)
    assert all(rec["template_idx"] < 8 for rec in records)
    assert "market_data" not in {rec["family"] for rec in records}


def test_tps_thresholds_are_inside_support_and_not_tau_grid_points():
    validate_thresholds()
    lo = min(TAU_VALUES_S)
    hi = max(TAU_VALUES_S)
    grid = set(TAU_VALUES_S)
    for family in FAMILIES:
        assert lo < family.threshold_s < hi
        assert family.threshold_s not in grid


def test_tps_monotonicity_uses_family_long_action_not_refresh():
    rows = []
    for tau, action in [(10, "REUSE"), (30, "REUSE"), (300, "ASK"), (1800, "ASK")]:
        rows.append({
            "condition": "hidden_only",
            "family": "safety_advice",
            "tau_ci_s": tau,
            "action": action,
        })
    result = monotonicity(rows)
    assert result["per_family"]["safety_advice"]["long_action"] == "ASK"
    assert result["r_log_tau_vs_p_long_action"] > 0.5


def test_track_b_training_data_cli_writes_explicit_output(tmp_path: Path):
    out = tmp_path / "tps_train.jsonl"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.tps.training_data",
            "--out",
            str(out),
            "--seed",
            "3",
            "--split",
            "train",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows
    assert all(row["track"] == "track_b_policy" for row in rows)
