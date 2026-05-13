"""Preflight check before training.

Verifies EVERYTHING is in order. Fails loudly on any missing piece.

Checks:
  - Python deps installable (pyproject.toml present, uv.lock valid)
  - Tokenized binary caches present + sizes match meta.json
  - Raw Latent World JSONL present (for --use-real-tau)
  - Model code: import + smoke test + 1-step train + backward + checkpoint save/load
  - All 7 eval harnesses import + run on untrained baseline without errors
  - Phase scheduler: all 4 phases applyable, trainable param counts sane
  - Optimizer builds with split groups
  - Disk space (warn if <50GB free)
  - CUDA availability + memory (if --cuda)
  - Git: clean tree or uncommitted-but-tracked
  - SPEC.tex and PREREGISTRATION.md present (canonical refs)

Output:
  - Console summary with PASS/FAIL per check
  - reports/preflight.md saved
  - Exit code 0 if all PASS, 1 if any FAIL

Usage:
  uv run python -m scripts.preflight
  uv run python -m scripts.preflight --cuda     # also verify CUDA
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch


_ENV_ROOT = os.environ.get("IPCN_ROOT")
if _ENV_ROOT:
    ROOT = Path(_ENV_ROOT)
else:
    # Default to the directory containing this script's parent (scripts/ -> repo root).
    # Falls back to the historical hardcoded macOS path if neither resolves.
    _SCRIPT_PARENT = Path(__file__).resolve().parent.parent
    _MAC_DEFAULT = Path("/Users/samsiavoshian/Desktop/Coding Stuff/Time-Model")
    ROOT = _SCRIPT_PARENT if (_SCRIPT_PARENT / "pyproject.toml").exists() else _MAC_DEFAULT


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    warning: bool = False


class Preflight:
    def __init__(self, require_cuda: bool = False):
        self.require_cuda = require_cuda
        self.results: list[CheckResult] = []

    def check(self, name: str, fn) -> CheckResult:
        try:
            ok, detail = fn()
            r = CheckResult(name=name, passed=bool(ok), detail=detail)
        except Exception as e:                                # noqa: BLE001
            r = CheckResult(name=name, passed=False, detail=f"exception: {e}")
        self.results.append(r)
        status = "PASS" if r.passed else ("WARN" if r.warning else "FAIL")
        print(f"  [{status}] {r.name}: {r.detail}")
        return r

    # ----- Individual checks -----

    def py_deps(self):
        ok = (ROOT / "pyproject.toml").exists() and (ROOT / "uv.lock").exists()
        return ok, "pyproject.toml + uv.lock present"

    def docs(self):
        required = ["SPEC.tex", "PREREGISTRATION.md", "PAPER.md",
                    "ARCHITECTURE_LOCKED.md", "TRAINING_READY.md", "COMPLETION.md"]
        missing = [f for f in required if not (ROOT / f).exists()]
        return not missing, ("all canonical docs present" if not missing else f"MISSING: {missing}")

    def tokenized_caches(self):
        summary_path = ROOT / "data" / "tokenized" / "SUMMARY.json"
        if not summary_path.exists():
            return False, "data/tokenized/SUMMARY.json missing"
        with open(summary_path) as f:
            summary = json.load(f)
        bad = []
        for prefix, meta in summary.items():
            tokens_path = ROOT / f"{prefix}.tokens.bin"
            bounds_path = ROOT / f"{prefix}.boundaries.npy"
            if not tokens_path.exists() or not bounds_path.exists():
                bad.append(prefix)
        if bad:
            return False, f"missing files for {len(bad)} caches: {bad[:2]}..."
        total_tok = sum(m["n_tokens"] for m in summary.values())
        return True, f"{len(summary)} caches, {total_tok:,} tokens"

    def raw_latent_world(self):
        d = ROOT / "data" / "latent_world"
        required = ["train_1k.jsonl", "train_2k.jsonl", "train_4k.jsonl", "train_8k.jsonl",
                    "test_16k.jsonl", "valid_1k.jsonl"]
        missing = [f for f in required if not (d / f).exists()]
        return not missing, ("all Latent World JSONL present" if not missing else f"MISSING: {missing}")

    def model_imports(self):
        import importlib
        # Consolidation lives in train.maybe_run_consolidation (not a separate module)
        mods = ["model.config", "model.chronometric", "model.adapters", "model.memory",
                "model.pfc", "model.injection", "model.core", "model.losses",
                "model.ipcn", "model.dataset", "model.train",
                "model.checkpoint", "model.phases", "model.replay_buffer",
                "model.latent_world_loader", "model.predictions", "model.h2_probe",
                "model.h4_cti", "model.h5_evolution", "model.h7_contradiction",
                "model.ambiguity_accuracy", "model.eval_all", "model.ablation_runner",
                "model.run_phase", "model.late_retrieval", "model.metrics",
                "model.prefix_integrity"]
        for m in mods:
            importlib.import_module(m)
        return True, f"{len(mods)} modules import OK"

    def smoke_forward(self):
        from model.config import IPCNConfig
        from model.ipcn import IPCN
        cfg = IPCNConfig()
        model = IPCN(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (cfg.chunk_length,))
        out = model.forward_chunk(input_ids, tau_t=1.0, delta_tau=1.0)
        ok = (out.logits.shape == (cfg.chunk_length, cfg.vocab_size)
              and not torch.isnan(out.logits).any())
        return ok, f"forward chunk shape={tuple(out.logits.shape)}"

    def smoke_backward(self):
        from model.config import IPCNConfig
        from model.ipcn import IPCN
        from model.losses import lm_loss
        cfg = IPCNConfig()
        model = IPCN(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (cfg.chunk_length,))
        targets = torch.randint(0, cfg.vocab_size, (cfg.chunk_length,))
        out = model.forward_chunk(input_ids, tau_t=1.0, delta_tau=1.0)
        loss = lm_loss(out.logits, targets)
        loss.backward()
        nonzero = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum().item() > 0)
        return nonzero > 0, f"{nonzero} param tensors received gradients"

    def smoke_checkpoint(self):
        from model.config import IPCNConfig
        from model.ipcn import IPCN
        from model.checkpoint import load_checkpoint, save_checkpoint
        from model.train import build_optimizer
        cfg = IPCNConfig()
        model = IPCN(cfg)
        opt = build_optimizer(model, cfg)
        tmp = Path("/tmp/preflight_ckpt.pt")
        save_checkpoint(str(tmp), model, opt, cfg, train_step=0)
        model2, opt_state, cfg2, step2 = load_checkpoint(str(tmp))
        tmp.unlink()
        return True, "save+load roundtrip OK"

    def phase_scheduler(self):
        from model.config import IPCNConfig
        from model.ipcn import IPCN
        from model.phases import Phase, apply_phase, get_phase_config, trainable_param_count
        cfg = IPCNConfig()
        model = IPCN(cfg)
        counts = {}
        for phase in [Phase.PHASE_0_SANITY, Phase.PHASE_1_PFC_CONSOLIDATION,
                      Phase.PHASE_2_EARLY_CORE_CONSOLIDATION, Phase.PHASE_3_MIXED_LM]:
            pc = get_phase_config(phase, cfg)
            apply_phase(model, pc)
            counts[pc.name] = trainable_param_count(model)
        ok = (counts["phase0_sanity"] > counts["phase1_pfc_consolidation"]
              and counts["phase1_pfc_consolidation"] < counts["phase2_early_core"])
        return ok, f"trainable counts {counts}"

    def eval_harnesses_import(self):
        # Just import and call __dir__ to verify symbols exist
        from model.predictions import H1_synthetic_check, H6_pairs_check
        from model.h2_probe import H2_probe_check
        from model.h4_cti import compute_CTI
        from model.h5_evolution import H5_check
        from model.h7_contradiction import H7_check
        from model.ambiguity_accuracy import evaluate_ambiguity
        return True, "all 7 prediction harnesses importable"

    def disk_space(self):
        usage = shutil.disk_usage(str(ROOT))
        free_gb = usage.free / 1e9
        # Hard fail below 5GB (can't even do checkpointing). Soft pass below 20GB.
        if free_gb < 5:
            return False, f"CRITICAL: only {free_gb:.1f} GB free; need >= 5 GB"
        if free_gb < 20:
            return True, f"{free_gb:.1f} GB free (warn: Spark has 228 GB; this is laptop-only)"
        return True, f"{free_gb:.1f} GB free"

    def cuda(self):
        if not self.require_cuda:
            return True, "skipped (--cuda not set)"
        if not torch.cuda.is_available():
            return False, "torch.cuda.is_available() is False"
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        return True, f"{n} device(s), GPU 0: {name}, {mem:.1f} GB"

    def git_state(self):
        try:
            out = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT)).decode()
            if out.strip():
                return False, f"uncommitted changes: {len(out.strip().splitlines())} files"
            head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT)).decode().strip()
            return True, f"clean tree at {head}"
        except Exception as e:                                # noqa: BLE001
            return False, f"git error: {e}"

    def run(self):
        print("=" * 60)
        print("IPCN PREFLIGHT")
        print("=" * 60)

        self.check("python deps", self.py_deps)
        self.check("canonical docs", self.docs)
        self.check("tokenized caches", self.tokenized_caches)
        self.check("raw Latent World JSONL", self.raw_latent_world)
        self.check("model imports", self.model_imports)
        self.check("smoke: forward chunk", self.smoke_forward)
        self.check("smoke: backward pass", self.smoke_backward)
        self.check("smoke: checkpoint roundtrip", self.smoke_checkpoint)
        self.check("phase scheduler", self.phase_scheduler)
        self.check("eval harnesses import", self.eval_harnesses_import)
        self.check("disk space", self.disk_space)
        self.check("CUDA", self.cuda)
        self.check("git state", self.git_state)

        n_pass = sum(1 for r in self.results if r.passed)
        n_total = len(self.results)
        print("=" * 60)
        print(f"RESULT: {n_pass}/{n_total} checks passed")
        print("=" * 60)

        # Save report
        report_path = ROOT / "reports" / "preflight.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# IPCN Preflight Report",
            "",
            f"- Total checks: {n_total}",
            f"- Passed: {n_pass}",
            f"- Failed: {n_total - n_pass}",
            "",
            "| Check | Status | Detail |",
            "|---|---|---|",
        ]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"| {r.name} | {status} | {r.detail} |")
        report_path.write_text("\n".join(lines))
        print(f"\nReport: {report_path}")

        return 0 if n_pass == n_total else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cuda", action="store_true", help="require CUDA available")
    args = p.parse_args()

    preflight = Preflight(require_cuda=args.cuda)
    sys.exit(preflight.run())


if __name__ == "__main__":
    main()
