from __future__ import annotations

import subprocess
import sys
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_shell_scripts_do_not_use_old_machine_paths():
    for path in (ROOT / "scripts").glob("*.sh"):
        text = path.read_text()
        assert "/home/omarramadan/ipcn" not in text
        assert "$HOME/Time-Model" not in text
        assert "$HOME/ipcn" not in text


def test_current_tps_artifacts_do_not_use_old_machine_paths():
    for path in (ROOT / "reports" / "tps").glob("*"):
        if path.suffix not in {".json", ".log", ".txt"}:
            continue
        text = path.read_text()
        assert "/home/omarramadan/ipcn" not in text, path


def test_supported_shell_runners_use_run_context_and_no_root_outputs():
    root_output = re.compile(r'^\s*(OUT|REC|REPORT|LOG|STDOUT|SENT|DATA|LOG_PATH|CKPT_PATH|REPORT_PATH)=["\']?(reports|logs|checkpoints|data)/', re.M)
    for path in (ROOT / "scripts").glob("*.sh"):
        text = path.read_text()
        assert "set -euo pipefail" in text
        assert "scripts/lib/run_context.sh" in text
        assert "time_model_init_run" in text
        assert not root_output.search(text), path


def test_supported_runner_requires_run_id_before_work():
    for script in [
        "run_external_bench.sh",
        "run_track_a_mechanistic.sh",
        "run_track_b_policy.sh",
        "run_track_c.sh",
    ]:
        proc = subprocess.run(
            ["bash", str(ROOT / "scripts" / script)],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert proc.returncode != 0, script
        assert "RUN_ID is required" in proc.stderr, script


def test_aggregate_refuses_implicit_legacy_glob():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "aggregate_seeds.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert "required" in proc.stderr


def test_tps_analyze_refuses_implicit_legacy_glob():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "tps" / "analyze.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert "required" in proc.stderr


def test_evaluators_do_not_default_to_archived_output_names():
    risky_defaults = [
        'default="reports/tpdr_results.json"',
        'default="reports/qwen_time_v11_probe.json"',
        'default="reports/qwen_time_v11_falsify.json"',
        'default="reports/qwen_time_v11_pressure.json"',
    ]
    for rel in [
        "eval/tpdr/run_tpdr.py",
        "model/qwen_time_probe.py",
        "model/qwen_time_falsify.py",
        "model/qwen_time_pressure.py",
    ]:
        text = (ROOT / rel).read_text()
        for default in risky_defaults:
            assert default not in text


def test_python_experiment_writers_have_no_root_output_defaults():
    forbidden = re.compile(
        r"default=[\"'](?:reports|logs|checkpoints|data)/|"
        r"Path\([\"'](?:reports|logs|checkpoints|data)/|"
        r"open\([\"'](?:reports|logs|checkpoints|data)/"
    )
    allow = {
        # Canonical manifest and paper/figure maintenance may read/write
        # tracked evidence or paper artifacts by design.
        "scripts/generate_current_manifest.py",
        "scripts/apply_tps_paper_edits.py",
        "scripts/build_probe_paper_update.py",
        "scripts/build_tpdr_paper_update.py",
        "scripts/build_tps_paper_update.py",
        "scripts/make_fig0_architecture.py",
        "scripts/make_fig5.py",
        "scripts/make_fig6.py",
        "scripts/make_fig7_t4_token_labeled.py",
        "scripts/make_fig8_dialogue.py",
        "scripts/make_fig9_cross_seed_variance.py",
        "scripts/make_fig10_scaling.py",
        "scripts/make_figures.py",
        "scripts/plot_probe_clock_heldout.py",
        "scripts/plot_tps_monotonicity.py",
        "scripts/agg_ablation_rows.py",
    }
    offenders = []
    for base in ["model", "eval", "scripts"]:
        for path in (ROOT / base).rglob("*.py"):
            rel = path.relative_to(ROOT).as_posix()
            if rel.startswith("scripts/legacy/") or rel in allow:
                continue
            text = path.read_text()
            if forbidden.search(text):
                offenders.append(rel)
    assert offenders == []


def test_direct_python_writers_require_output_scope():
    cases = [
        (
            [sys.executable, "-m", "model.qwen_time_check", "--device", "cpu"],
            "output scope required",
        ),
        (
            [sys.executable, "-m", "model.qwen_time_train", "--data", "missing.jsonl", "--device", "cpu"],
            "output scope required",
        ),
        (
            [sys.executable, "eval/tpdr/analyze_v2.py"],
            "output scope required",
        ),
        (
            [sys.executable, "-m", "eval.external.eval_tau_bench", "--adapter", "vanilla"],
            "output scope required",
        ),
        (
            [sys.executable, "-m", "eval.tps.classifier_baseline"],
            "output scope required",
        ),
        (
            [sys.executable, "-m", "eval.tps.benchmark"],
            "output scope required",
        ),
        (
            [sys.executable, "-m", "eval.tps.training_data"],
            "output scope required",
        ),
        (
            [sys.executable, "-m", "eval.track_c.generate", "--seed", "0"],
            "output scope required",
        ),
        (
            [sys.executable, "-m", "eval.track_c.baselines", "--train", "missing.jsonl", "--eval", "missing.jsonl"],
            "output scope required",
        ),
        (
            [sys.executable, "-m", "eval.external.generate_tau_sessions"],
            "output scope required",
        ),
        (
            [sys.executable, "scripts/bootstrap_existing.py"],
            "output scope required",
        ),
    ]
    for cmd, expected in cases:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        assert proc.returncode != 0, cmd
        assert expected in (proc.stderr + proc.stdout), (cmd, proc.stderr, proc.stdout)


def test_legacy_scripts_are_not_current_generators():
    manifest = (ROOT / "reports" / "current" / "manifest.json").read_text()
    assert "scripts/legacy/" not in manifest
    assert (ROOT / "scripts" / "legacy" / "README.md").exists()


def test_only_current_docs_remain_at_repo_root():
    allowed = {"README.md", "REPRODUCIBILITY.md", "ARTIFACT.md"}
    root_docs = {p.name for p in ROOT.iterdir() if p.is_file() and p.suffix in {".md", ".tex"}}
    assert root_docs == allowed
