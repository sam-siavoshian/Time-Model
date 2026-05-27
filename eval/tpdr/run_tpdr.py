"""Run TPDR benchmark on vanilla / prompt / ci adapters.
50 scenarios x N tau values x 3 adapters.
For each (scenario, tau, adapter) run greedy decode and compute metrics.
Then report:
  per-scenario tau elasticity for each metric and adapter
  pooled correlations across all 50 scenarios
  CI vs prompt vs vanilla on every metric
Output: reports/tpdr_results.json
"""
from __future__ import annotations
import argparse, json, math, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer
from model.qwen_time import QwenTime, QwenTimeConfig, build_qwen_time
from eval.tpdr.scenarios import get_scenarios
from eval.tpdr.metrics import all_metrics, pearson_safe


# MPS-safe injector patch
import torch as _torch_patch
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


def tau_text(tau):
    if tau < 60: return f"[elapsed: {tau:.1f}s]"
    if tau < 3600: return f"[elapsed: {tau/60:.1f}m]"
    if tau < 86400:
        h = int(tau // 3600); m = int((tau % 3600) // 60)
        return f"[elapsed: {h}h {m}m]"
    d = int(tau // 86400); h = int((tau % 86400) // 3600)
    return f"[elapsed: {d}d {h}h]"


@torch.no_grad()
def greedy_decode(model, tok, prompt, tau, max_new=200, device="cuda"):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = ids.clone()
    eos = tok.eos_token_id
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    for _ in range(max_new):
        result = model(out, tau_t=float(tau))
        logits = result["logits"] if isinstance(result, dict) else result.logits
        if logits.dim() == 2: logits = logits.unsqueeze(0)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        out = torch.cat([out, nxt], dim=1)
        if nxt.item() in (eos, im_end): break
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def load_ci(ckpt, base, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg_d = ck.get("cfg", {})
    if not isinstance(cfg_d, dict): cfg_d = vars(cfg_d)
    valid = set(QwenTimeConfig.__dataclass_fields__.keys())
    cfg_kwargs = {k: v for k, v in cfg_d.items() if k in valid}
    cfg_kwargs["base_model_name"] = base
    cfg = QwenTimeConfig(**cfg_kwargs)
    model = build_qwen_time(cfg)
    model.load_state_dict(ck["trainable_state"], strict=False)
    model.eval()
    return model.to(device)


def load_vanilla(base, device):
    """Vanilla = same architecture, alpha frozen at zero. So chrono channel exists but is identity."""
    cfg = QwenTimeConfig()
    cfg.base_model_name = base
    model = build_qwen_time(cfg)
    # alphas are already zero by init; freeze them and we don't load any state
    model.eval()
    return model.to(device)


def format_prompt(scenario, adapter, tau):
    """For CI and vanilla: just the scenario. For prompt: tau-prefixed."""
    if adapter == "prompt":
        msg = f"{tau_text(tau)} {scenario}"
    else:
        msg = scenario
    return (f"<|im_start|>user\n{msg}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--ci-ckpt", default="release_ckpts/qwen_time_v15s_20260523_141410_seed0.pt")
    ap.add_argument("--prompt-ckpt", default="",
                    help="Optional: trained prompt-baseline checkpoint (LoRA only, freeze-alpha).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-tau", type=int, default=10)
    ap.add_argument("--n-scenarios", type=int, default=50)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--out", default="reports/tpdr_results.json")
    args = ap.parse_args()

    scenarios = get_scenarios()[:args.n_scenarios]
    # log-uniform tau grid
    log_t_lo = math.log(1.0); log_t_hi = math.log(7 * 86400.0)
    taus = [math.exp(log_t_lo + i * (log_t_hi - log_t_lo) / (args.n_tau - 1))
            for i in range(args.n_tau)]
    print(f"TPDR: {len(scenarios)} scenarios x {len(taus)} tau values = {len(scenarios)*len(taus)} per adapter")
    print(f"tau range: [{taus[0]:.1f}, {taus[-1]:.1f}]")

    tok = AutoTokenizer.from_pretrained(args.base)

    out = {"taus": taus, "n_scenarios": len(scenarios), "scenarios": scenarios, "adapters": {}}

    # adapter specs (load sequentially to fit 16GB RAM)
    adapter_specs = [("vanilla", "vanilla", None, None),
                     ("ci", "ci", args.ci_ckpt, None)]
    if args.prompt_ckpt and Path(args.prompt_ckpt).exists():
        adapter_specs.append(("prompt", "prompt", args.prompt_ckpt, "prompt"))
    else:
        adapter_specs.append(("prompt", "vanilla", None, "prompt"))

    for adapter_name, adapter_type, ckpt, prompt_mode in adapter_specs:
        print(f"\n[setup] loading {adapter_name} ({adapter_type})...")
        if adapter_type == "vanilla":
            model = load_vanilla(args.base, args.device)
        else:
            model = load_ci(ckpt, args.base, args.device)

        print(f"\n=== adapter: {adapter_name} ===")
        per_scenario_results = []
        for s_i, scenario in enumerate(scenarios):
            per_tau_metrics = []
            for tau in taus:
                # adapter='prompt' or prompt_mode='prompt' both inject in text
                effective_adapter = "prompt" if (adapter_name == "prompt" or prompt_mode == "prompt") else adapter_name
                prompt_text = format_prompt(scenario, effective_adapter, tau)
                resp = greedy_decode(model, tok, prompt_text, tau, max_new=args.max_new, device=args.device)
                m = all_metrics(resp)
                m["response"] = resp[:300]  # truncated for storage
                m["tau"] = tau
                per_tau_metrics.append(m)
            per_scenario_results.append({
                "scenario_idx": s_i,
                "metrics_per_tau": per_tau_metrics,
            })
            if (s_i + 1) % 5 == 0:
                # print elasticity for length on this scenario
                ls = [m["length_words"] for m in per_tau_metrics]
                logtaus = [math.log(t) for t in taus]
                r = pearson_safe(logtaus, ls)
                print(f"  scenario {s_i+1}/{len(scenarios)}: length elasticity r={r:.3f}")

        # Aggregate across all scenarios
        agg = {"per_metric_elasticity": {}}
        for key in ["length_chars", "length_words", "urgency_score",
                    "deliberative_score", "hedge_score", "conditional_clauses",
                    "imperative_count"]:
            rs = []
            for sc in per_scenario_results:
                vals = [m[key] for m in sc["metrics_per_tau"]]
                logtaus = [math.log(t) for t in taus]
                r = pearson_safe(logtaus, vals)
                if not math.isnan(r):
                    rs.append(r)
            if rs:
                from statistics import mean, stdev
                agg["per_metric_elasticity"][key] = {
                    "mean_r": mean(rs), "std_r": stdev(rs) if len(rs) > 1 else 0.0,
                    "n_scenarios_with_finite_r": len(rs),
                    "median_r": sorted(rs)[len(rs)//2],
                }
        out["adapters"][adapter_name] = {
            "per_scenario_results": per_scenario_results,
            "aggregate": agg,
        }
        print(f"  AGGREGATE {adapter_name}:")
        for k, v in agg["per_metric_elasticity"].items():
            print(f"    {k:24} mean_r={v['mean_r']:+.3f}  std={v['std_r']:.3f}  n={v['n_scenarios_with_finite_r']}")
        # save partial state and unload to free memory
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2, default=str))
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        elif hasattr(torch, "mps"): 
            try: torch.mps.empty_cache()
            except: pass
        import gc; gc.collect()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
