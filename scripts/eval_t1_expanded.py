"""W2: T1 effective-n expansion to 50 unique tau + bootstrap CI.
Loads each cross-seed checkpoint, runs T1 on 50 log-uniform tau,
reports Pearson r + bootstrap 95% CI per seed.
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
def greedy(model, tok, prompt, tau, max_new=24, device="mps"):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = ids.clone()
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eos = tok.eos_token_id
    for _ in range(max_new):
        out_dict = model(out, tau_t=float(tau)); logits = out_dict["logits"]; 
        if logits.dim() == 2: logits = logits.unsqueeze(0)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        out = torch.cat([out, nxt], dim=1)
        if nxt.item() in (eos, im_end): break
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def boot_r(true, pred, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    true = np.array(true); pred = np.array(pred)
    valid = ~(np.isnan(true) | np.isnan(pred))
    true = true[valid]; pred = pred[valid]
    n = len(true)
    if n < 3: return float("nan"), float("nan"), float("nan")
    base_r = float(np.corrcoef(true, pred)[0, 1])
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = true[idx]; b = pred[idx]
        if np.std(a) == 0 or np.std(b) == 0: continue
        rs.append(np.corrcoef(a, b)[0, 1])
    rs = np.array(rs)
    return base_r, float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n-tau", type=int, default=50)
    ap.add_argument("--ckpt-glob", default="release_ckpts/qwen_time_v15s_*seed*.pt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.base)
    rng = random.Random(2026)
    taus = [math.exp(rng.uniform(math.log(1.0), math.log(7 * 86400.0)))
            for _ in range(args.n_tau)]
    prompt = "<|im_start|>user\nHow long has it been since we started?<|im_end|>\n<|im_start|>assistant\n"
    import glob
    ckpts = sorted(glob.glob(args.ckpt_glob))
    print(f"found {len(ckpts)} ckpts")
    out = {"n_tau": args.n_tau, "taus": taus, "seeds": {}}
    for cp in ckpts:
        seed_id = re.search(r"seed(\d+)", cp).group(1)
        print(f"\n=== seed{seed_id} ===")
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        cfg_d = ck.get("cfg", {})
        if not isinstance(cfg_d, dict): cfg_d = vars(cfg_d)
        valid_keys = set(QwenTimeConfig.__dataclass_fields__.keys())
        cfg_kwargs = {k: v for k, v in cfg_d.items() if k in valid_keys}
        cfg_kwargs["base_model_name"] = args.base
        cfg = QwenTimeConfig(**cfg_kwargs)
        model = build_qwen_time(cfg)
        model.load_state_dict(ck["trainable_state"], strict=False)
        model.eval()
        model = model.to(args.device)
        parsed = []
        for i, t in enumerate(taus):
            resp = greedy(model, tok, prompt, t, device=args.device)
            p = parse_duration(resp); parsed.append(p)
            if i < 6 or i % 10 == 0:
                print(f"  [{i}] tau={t:.1f}  {resp!r}  parsed={p}")
        base_r, lo, hi = boot_r(taus, parsed)
        out["seeds"][f"seed{seed_id}"] = {"parsed": parsed,
                                          "pearson_r": base_r,
                                          "bootstrap_95_lo": lo,
                                          "bootstrap_95_hi": hi}
        print(f"  seed{seed_id}: r={base_r:.4f}  95%CI=[{lo:.4f}, {hi:.4f}]  effective_n={sum(1 for p in parsed if not math.isnan(p))}")
        del model
        torch.mps.empty_cache() if hasattr(torch, "mps") else None
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
