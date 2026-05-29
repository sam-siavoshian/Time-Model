"""Stripped TPDR runner for the vanilla-baseline diagnostic ONLY.

Pre-registered in docs/experiments/current/PREREGISTRATION_v2.md section 1.8
as configuration (b).

Differences from run_tpdr.py:
  - No chat template wrapping. The model receives the raw scenario text
    followed by a single newline.
  - No system prompt.
  - Only the vanilla adapter is loaded (CI/prompt are irrelevant to the
    diagnostic).
  - Output schema matches run_tpdr.py for direct comparison.

This script is touched ONLY by the diagnostic and is not part of the
pre-registered main sweep.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer
from model.qwen_time import QwenTime, QwenTimeConfig, build_qwen_time
from eval.tpdr.scenarios import get_scenarios
from eval.tpdr.metrics import all_metrics, pearson_safe

from model.qwen_time import _ChronoInjector as _CI
def _patched_forward(self, h, chi_t):
    chi_x = chi_t.to(self.to_gamma.weight.dtype)
    gamma = self.to_gamma(chi_x); beta = self.to_beta(chi_x)
    a = self.alpha[None, None, :].to(h.dtype)
    ge = gamma[None, None, :].to(h.dtype); be = beta[None, None, :].to(h.dtype)
    if self.injection_type == "additive":
        return h + a * be
    return h + a * (ge * h + be)
_CI.forward = _patched_forward


@torch.no_grad()
def greedy_decode(model, tok, prompt, tau, max_new=150, device="cuda"):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = ids.clone()
    eos = tok.eos_token_id
    for _ in range(max_new):
        result = model(out, tau_t=float(tau))
        logits = result["logits"] if isinstance(result, dict) else result.logits
        if logits.dim() == 2: logits = logits.unsqueeze(0)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        out = torch.cat([out, nxt], dim=1)
        if nxt.item() == eos: break
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def load_vanilla(base, device):
    cfg = QwenTimeConfig()
    cfg.base_model_name = base
    model = build_qwen_time(cfg)
    model.train(False)
    return model.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-tau", type=int, default=10)
    ap.add_argument("--n-scenarios", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=150)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scenarios = get_scenarios()[:args.n_scenarios]
    log_t_lo = math.log(1.0); log_t_hi = math.log(7 * 86400.0)
    taus = [math.exp(log_t_lo + i * (log_t_hi - log_t_lo) / (args.n_tau - 1))
            for i in range(args.n_tau)]

    tok = AutoTokenizer.from_pretrained(args.base)
    model = load_vanilla(args.base, args.device)

    out = {"taus": taus, "n_scenarios": len(scenarios), "scenarios": scenarios,
           "config": "stripped: raw prompt, no chat template, no system prompt",
           "adapters": {"vanilla_stripped": {"per_scenario_results": []}}}

    for s_i, scenario in enumerate(scenarios):
        per_tau_metrics = []
        for tau in taus:
            prompt_text = f"{scenario}\n"
            resp = greedy_decode(model, tok, prompt_text, tau,
                                 max_new=args.max_new, device=args.device)
            m = all_metrics(resp)
            m["response"] = resp[:300]
            m["tau"] = tau
            per_tau_metrics.append(m)
        out["adapters"]["vanilla_stripped"]["per_scenario_results"].append({
            "scenario_idx": s_i, "metrics_per_tau": per_tau_metrics,
        })
        if (s_i + 1) % 2 == 0:
            print(f"  [{s_i+1}/{len(scenarios)}] resp@tau={taus[0]:.1f}: "
                  f"{per_tau_metrics[0]['response'][:80]!r}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
