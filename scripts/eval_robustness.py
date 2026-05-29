"""W15: how robust is CI to wrong / noisy / missing tau?
Conditions: A clean, B noisy log10(tau), C random tau, D tau=0, E tau=1e9.
Output: pass --out explicitly, preferably under runs/<run_id>/reports/.
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
def greedy(model, tok, prompt: str, tau: float, max_new: int = 24,
           device: str = "mps") -> str:
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = ids.clone()
    eos = tok.eos_token_id
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    for _ in range(max_new):
        out_dict = model(out, tau_t=float(tau)); logits = out_dict["logits"]; 
        if logits.dim() == 2: logits = logits.unsqueeze(0)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        out = torch.cat([out, nxt], dim=1)
        if nxt.item() in (eos, im_end): break
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def pearson(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    if len(pairs) < 2: return float("nan")
    a_ = np.array([p[0] for p in pairs]); b_ = np.array([p[1] for p in pairs])
    if np.std(a_) == 0 or np.std(b_) == 0: return float("nan")
    return float(np.corrcoef(a_, b_)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="release_ckpts/qwen_time_v15s_20260523_141410_seed0.pt")
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n-tau", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print(f"loading {args.base}...")
    tok = AutoTokenizer.from_pretrained(args.base)
    print(f"loading ckpt {args.ckpt}...")
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
    print("model loaded.")
    rng = random.Random(2026)
    true_taus = [math.exp(rng.uniform(math.log(1.0), math.log(7 * 86400.0)))
                 for _ in range(args.n_tau)]
    prompt = "<|im_start|>user\nHow long has it been since we started?<|im_end|>\n<|im_start|>assistant\n"
    results = {}
    print("\nA. clean baseline:")
    parsed_A = []
    for tau in true_taus:
        resp = greedy(model, tok, prompt, tau, device=args.device)
        p = parse_duration(resp); parsed_A.append(p)
        print(f"  tau={tau:.1f}  -> {resp!r}  -> {p}")
    rA = pearson(true_taus, parsed_A)
    results["A_clean"] = {"true_taus": true_taus, "parsed": parsed_A, "pearson_r": rA}
    print(f"  A r = {rA:.4f}")
    print("\nB. log10(tau) + Gaussian noise:")
    for sigma in [0.1, 0.3, 1.0]:
        parsed = []; noisy_taus = []
        for tau in true_taus:
            log_n = math.log10(tau) + rng.gauss(0, sigma)
            ntau = max(0.1, min(7 * 86400 * 10, 10 ** log_n))
            noisy_taus.append(ntau)
            parsed.append(parse_duration(greedy(model, tok, prompt, ntau, device=args.device)))
        r_t = pearson(true_taus, parsed); r_n = pearson(noisy_taus, parsed)
        results[f"B_sigma_{sigma}"] = {"noisy_taus": noisy_taus, "parsed": parsed,
                                       "pearson_r_vs_true": r_t,
                                       "pearson_r_vs_noisy_input": r_n}
        print(f"  sigma={sigma}: r_vs_true={r_t:.4f}, r_vs_noisy={r_n:.4f}")
    print("\nC. random tau:")
    random_taus = [math.exp(rng.uniform(math.log(1.0), math.log(7 * 86400.0)))
                   for _ in range(args.n_tau)]
    parsed_C = [parse_duration(greedy(model, tok, prompt, t, device=args.device))
                for t in random_taus]
    rC = pearson(true_taus, parsed_C); rC_in = pearson(random_taus, parsed_C)
    results["C_random_tau"] = {"random_taus": random_taus, "parsed": parsed_C,
                                "pearson_r_vs_unrelated_true": rC,
                                "pearson_r_vs_random_input": rC_in}
    print(f"  C: r_vs_true={rC:.4f}, r_vs_random_input={rC_in:.4f}")
    print("\nD. tau=0:")
    parsed_D = [parse_duration(greedy(model, tok, prompt, 0.0, device=args.device))
                for _ in range(args.n_tau)]
    unique_D = list({x for x in parsed_D})
    print(f"  D: {len(unique_D)} unique parsed values: {unique_D[:5]}")
    results["D_tau_zero"] = {"parsed": parsed_D, "unique_count": len(unique_D)}
    print("\nE. tau=1e9 adversarial:")
    parsed_E = [parse_duration(greedy(model, tok, prompt, 1e9, device=args.device))
                for _ in range(args.n_tau)]
    unique_E = list({x for x in parsed_E})
    print(f"  E: {len(unique_E)} unique parsed values: {unique_E[:5]}")
    results["E_tau_1e9"] = {"parsed": parsed_E, "unique_count": len(unique_E)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
