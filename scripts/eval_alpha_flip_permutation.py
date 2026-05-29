"""W5 + Q6: permutation test for half-layer alpha-sign-flip.
For seed-0 checkpoint, run 50 random 17-of-35 subset flips and report
the distribution of resulting Pearson r values. Also runs the targeted
top-8 / bottom-8 / random-8 flips on seeds 1 and 2 to test cross-seed
replication of mid-deep dominance.
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


def eval_t1(model, tok, taus, device):
    prompt = "<|im_start|>user\nHow long has it been since we started?<|im_end|>\n<|im_start|>assistant\n"
    parsed = []
    for t in taus:
        parsed.append(parse_duration(greedy(model, tok, prompt, t, device=device)))
    pairs = [(t, p) for t, p in zip(taus, parsed) if not math.isnan(p)]
    if len(pairs) < 3: return float("nan"), parsed
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    if np.std(a) == 0 or np.std(b) == 0: return float("nan"), parsed
    return float(np.corrcoef(a, b)[0, 1]), parsed


def load_model(ckpt, base, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg_d = ck.get("cfg", {})
    if not isinstance(cfg_d, dict): cfg_d = vars(cfg_d)
    valid = set(QwenTimeConfig.__dataclass_fields__.keys())
    cfg_kwargs = {k: v for k, v in cfg_d.items() if k in valid}
    cfg_kwargs["base_model_name"] = base
    cfg = QwenTimeConfig(**cfg_kwargs)
    model = build_qwen_time(cfg)
    model.load_state_dict(ck["trainable_state"], strict=False)
    pass  # use default dtype
    model.eval()
    return model.to(device)


def get_alpha_layers(model):
    """Return list of (layer_idx, alpha_param) pairs."""
    out = []
    for name, p in model.named_parameters():
        m = re.search(r"chrono_injectors\.(\d+)\.alpha", name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda kv: kv[0])
    return out


@torch.no_grad()
def with_flipped_alphas(model, layer_set, taus, tok, device):
    alphas = get_alpha_layers(model)
    # save originals
    orig = {li: p.data.clone() for li, p in alphas}
    try:
        for li, p in alphas:
            if li in layer_set:
                p.data.mul_(-1)
        r, parsed = eval_t1(model, tok, taus, device)
    finally:
        for li, p in alphas:
            p.data.copy_(orig[li])
    return r, parsed


def topk_layers(model, k, top=True):
    alphas = get_alpha_layers(model)
    norms = [(li, float(p.abs().mean().item())) for li, p in alphas]
    norms.sort(key=lambda kv: -kv[1] if top else kv[1])
    return [kv[0] for kv in norms[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n-tau", type=int, default=6)
    ap.add_argument("--n-permutations", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.base)
    rng = random.Random(2026)
    taus = [math.exp(rng.uniform(math.log(1.0), math.log(7 * 86400.0)))
            for _ in range(args.n_tau)]
    out = {"n_tau": args.n_tau, "taus": taus, "seeds": {}}
    for seed_id in [0, 1, 2]:
        cp = f"release_ckpts/qwen_time_v15s_20260523_141410_seed{seed_id}.pt"
        print(f"\n=== seed{seed_id} ===")
        model = load_model(cp, args.base, args.device)
        all_layers = [li for li, _ in get_alpha_layers(model)]
        n = len(all_layers); half = n // 2
        seed_out = {"layer_indices": all_layers, "n_layers": n}
        # baseline
        r_base, _ = eval_t1(model, tok, taus, args.device)
        seed_out["baseline_r"] = r_base
        print(f"  baseline r = {r_base:.4f}")
        # all-layer flip
        r_all, _ = with_flipped_alphas(model, set(all_layers), taus, tok, args.device)
        seed_out["all_flip_r"] = r_all
        print(f"  all-flip r = {r_all:.4f}")
        # top-8 / bot-8 / rand-8
        top8 = topk_layers(model, 8, top=True)
        bot8 = topk_layers(model, 8, top=False)
        r_top8, _ = with_flipped_alphas(model, set(top8), taus, tok, args.device)
        r_bot8, _ = with_flipped_alphas(model, set(bot8), taus, tok, args.device)
        seed_out["top8_layers"] = top8
        seed_out["bot8_layers"] = bot8
        seed_out["top8_flip_r"] = r_top8
        seed_out["bot8_flip_r"] = r_bot8
        print(f"  top-8 ({top8}) flip r = {r_top8:.4f}")
        print(f"  bot-8 ({bot8}) flip r = {r_bot8:.4f}")
        # random-8 control (3 samples for sanity)
        rand8_rs = []
        for trial in range(3):
            r8 = random.Random(trial * 7 + seed_id).sample(all_layers, 8)
            r, _ = with_flipped_alphas(model, set(r8), taus, tok, args.device)
            rand8_rs.append({"layers": r8, "r": r})
        seed_out["random8_trials"] = rand8_rs
        print(f"  random-8 trials: {[t['r'] for t in rand8_rs]}")
        # permutation test (only on seed 0 by default to save time)
        if seed_id == 0:
            print(f"  permutation test ({args.n_permutations} random half-subsets)...")
            perm_rs = []
            for k in range(args.n_permutations):
                sub = random.Random(k * 13 + 1).sample(all_layers, half)
                r, _ = with_flipped_alphas(model, set(sub), taus, tok, args.device)
                perm_rs.append({"subset": sub, "r": r})
                if (k + 1) % 10 == 0:
                    rs = [p["r"] for p in perm_rs if not math.isnan(p["r"])]
                    print(f"    [{k+1}/{args.n_permutations}] median r = {np.median(rs):.3f}  range=[{min(rs):.3f}, {max(rs):.3f}]")
            seed_out["permutation_test"] = perm_rs
        out["seeds"][f"seed{seed_id}"] = seed_out
        del model
        if hasattr(torch, "mps"):
            torch.mps.empty_cache()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
