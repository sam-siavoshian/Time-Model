"""W14: T1/T1b under temperature sampling (vs greedy headline). For each
tau condition, sample 20 outputs at temperature 0.7. Report within-condition
mean parsed-tau + std, then Pearson r over the 24 tau bins each with 20
samples. Tests whether the headline T1 r survives stochastic decoding.
"""
from __future__ import annotations
import argparse, json, math, random, re, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.qwen_time import QwenTime, QwenTimeConfig, build_qwen_time
# MPS-safe: override the chrono injector forward to keep dtype homogeneous,
# avoiding the MPS-bf16 matmul accumulator error on Apple Silicon.
import torch as _torch_for_patch
from model.qwen_time import _ChronoInjector as _CI
def _patched_forward(self, h, chi_t):
    chi_x = chi_t.to(self.to_gamma.weight.dtype)
    gamma = self.to_gamma(chi_x)
    beta = self.to_beta(chi_x)
    h_d = h
    a = self.alpha[None, None, :].to(h_d.dtype)
    gamma_e = gamma[None, None, :].to(h_d.dtype)
    beta_e = beta[None, None, :].to(h_d.dtype)
    if self.injection_type == "additive":
        out = h_d + a * beta_e
    else:
        modulated = gamma_e * h_d + beta_e
        out = h_d + a * modulated
    return out
_CI.forward = _patched_forward



def parse_duration(s: str) -> float:
    s = s.lower()
    for pat, scale in [(r"(\d+)\s*second", 1), (r"(\d+)\s*minute", 60),
                       (r"(\d+)\s*hour", 3600), (r"(\d+)\s*day", 86400)]:
        m = re.search(pat, s)
        if m: return float(m.group(1)) * scale
    return float("nan")


@torch.no_grad()
def sample(model, tok, prompt, tau, temperature=0.7, max_new=24, device="mps"):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = ids.clone()
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eos = tok.eos_token_id
    for _ in range(max_new):
        out_dict = model(out, tau_t=float(tau))
        logits = out_dict["logits"]
        if logits.dim() == 2: logits = logits.unsqueeze(0)
        logits = logits[:, -1] / temperature
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        out = torch.cat([out, nxt], dim=1)
        if nxt.item() in (eos, im_end): break
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="release_ckpts/qwen_time_v15s_20260523_141410_seed0.pt")
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n-tau", type=int, default=24)
    ap.add_argument("--n-sample-per-tau", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.base)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_d = ck.get("cfg", {})
    if not isinstance(cfg_d, dict): cfg_d = vars(cfg_d)
    valid = set(QwenTimeConfig.__dataclass_fields__.keys())
    cfg_kwargs = {k: v for k, v in cfg_d.items() if k in valid}
    cfg_kwargs["base_model_name"] = args.base
    cfg = QwenTimeConfig(**cfg_kwargs)
    model = build_qwen_time(cfg)
    model.load_state_dict(ck["trainable_state"], strict=False)
    pass  # use default dtype
    model.eval()
    model = model.to(args.device)
    rng = random.Random(2026)
    taus = [math.exp(rng.uniform(math.log(1.0), math.log(7 * 86400.0)))
            for _ in range(args.n_tau)]
    prompt = "<|im_start|>user\nHow long has it been since we started?<|im_end|>\n<|im_start|>assistant\n"
    out = {"taus": taus, "temperature": args.temperature,
           "n_sample_per_tau": args.n_sample_per_tau, "per_tau": []}
    all_pairs = []
    for i, t in enumerate(taus):
        per_tau_parsed = []
        per_tau_resp = set()
        for _ in range(args.n_sample_per_tau):
            torch.manual_seed(random.randint(0, 1_000_000))
            resp = sample(model, tok, prompt, t, args.temperature, device=args.device)
            per_tau_resp.add(resp)
            p = parse_duration(resp)
            per_tau_parsed.append(p)
            if not math.isnan(p):
                all_pairs.append((t, p))
        valid = [p for p in per_tau_parsed if not math.isnan(p)]
        out["per_tau"].append({
            "tau": t, "parsed": per_tau_parsed,
            "mean_parsed": float(np.mean(valid)) if valid else float("nan"),
            "std_parsed": float(np.std(valid)) if valid else float("nan"),
            "n_unique_responses": len(per_tau_resp),
        })
        if i < 5 or i % 5 == 0:
            print(f"  tau={t:.1f}  n_unique={len(per_tau_resp)}  mean={out['per_tau'][-1]['mean_parsed']}")
    arr = np.array(all_pairs)
    if len(arr) > 2:
        r = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
    else:
        r = float("nan")
    out["overall_pearson_r"] = r
    out["total_samples"] = len(all_pairs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"\noverall r = {r:.4f}  on {len(all_pairs)} samples ({args.n_tau} tau x {args.n_sample_per_tau} samples)")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
