from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_paper_claims.py"


def load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_paper_claims", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_claim_audit_all_passes_current_paper():
    run_id = f"pytest_claim_audit_{os.getpid()}"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--run-id", run_id, "--all"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (ROOT / "runs" / run_id / "reports" / "claim_audit.json").exists()
    assert (ROOT / "runs" / run_id / "reports" / "claim_audit.md").exists()


def test_claim_audit_random_sample_is_deterministic():
    audit = load_audit_module()
    first = audit.run_audit(seed=123, sample_size=4, include_all=False)
    second = audit.run_audit(seed=123, sample_size=4, include_all=False)
    assert first["selection"]["audited_claim_ids"] == second["selection"]["audited_claim_ids"]
    assert first["selection"]["audited_claim_ids"][: len(audit.SENTINEL_IDS)] == audit.SENTINEL_IDS


def test_claim_audit_detects_known_stale_patterns():
    audit = load_audit_module()
    base = (ROOT / "paper" / "main.tex").read_text()
    stale_text = base + r"""

The stale encoder says \omega_k=2\pi/T_k.
The stale external benchmark says CI adaptive length has r=+0.184.
The stale TPDR v1 claim says p=2.91e-3.
Half-layer flips & 52 preserve, 18 invert, 23 mixed, 7 nonfinite.
"""
    payload = audit.run_audit(paper_text=stale_text, include_all=True)
    verdicts = {item["claim_id"]: item["verdict"] for item in payload["items"]}
    assert verdicts["encoder_formula_no_2pi"] == "stale"
    assert verdicts["tau_sessions_negative"] == "stale"
    assert verdicts["tpdr_v2_headline"] == "stale"
    assert verdicts["half_layer_flip_counts"] == "ambiguous"
    assert not payload["ok"]


def test_claim_audit_requires_run_id():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--all"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert "--run-id" in proc.stderr
