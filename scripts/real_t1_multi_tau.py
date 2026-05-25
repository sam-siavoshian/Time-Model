"""Generate REAL T1 clock-readout decodes at multiple tau values for Figure 8.

Loads a released v15 cross-seed checkpoint and runs greedy decoding on the
clock prompt at six tau values spanning seconds to days. Dumps the actual
model outputs to reports/real_t1_multi_tau.json so the dialogue figure can
quote real model text instead of a synthesized one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model.qwen_time import QwenTimeConfig, build_qwen_time
from model.qwen_time_check import load_trainable, greedy_decode


PROMPT = ("<|im_start|>user\nHow long has it been since we started?"
          "<|im_end|>\n<|im_start|>assistant\n")

DEFAULT_TAUS = [5.0, 60.0, 600.0, 3600.0, 21600.0, 86400.0]
V15_TIMESCALES = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096,
                  16384, 65536, 86400, 604800)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--device", default="mps")
    p.add_argument("--out", default="reports/real_t1_multi_tau.json")
    p.add_argument("--max-new", type=int, default=24)
    args = p.parse_args()

    cfg = QwenTimeConfig()
    cfg.base_model_name = args.base
    cfg.timescales = V15_TIMESCALES
    print(f"loading base {args.base}...")
    model = build_qwen_time(cfg)
    print(f"moving to {args.device}...")
    model = model.to(args.device)
    print(f"loading trainables from {args.checkpoint}...")
    load_trainable(model, args.checkpoint)
    model.train(False)

    out = []
    for tau in DEFAULT_TAUS:
        print(f"\n=== tau = {tau} s ===")
        text = greedy_decode(model, PROMPT, tau, max_new=args.max_new,
                             device=args.device)
        clean = text.split("<|im_end|>")[0].strip()
        print(f"  -> {clean!r}")
        out.append({"tau_seconds": tau, "response": clean})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "prompt": "How long has it been since we started?",
        "checkpoint": args.checkpoint,
        "base": args.base,
        "decoding": "greedy, max_new=" + str(args.max_new),
        "samples": out,
    }, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
