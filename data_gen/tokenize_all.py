"""Tokenize all generated JSONL datasets to binary caches.

For each (input.jsonl, output_prefix) job:
  1. Iterate JSONL examples.
  2. Render each example to canonical text via renderers.py.
  3. Tokenize with GPT-2 BPE (tiktoken).
  4. Append token IDs to a flat uint16 array.
  5. Record (start, end) boundaries per example.
  6. Save:
       <prefix>.tokens.bin    flat uint16 array (raw bytes)
       <prefix>.boundaries.npy int64 (N, 2) array
       <prefix>.meta.json     summary stats

Total expected output: ~700-900 MB of tokens (much less than 3 GB JSONL).

Usage:
  uv run python -m data_gen.tokenize_all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tiktoken
from tqdm import tqdm

from data_gen import renderers


# Each job: (input_jsonl, output_prefix, dataset_type, renderer_fn_or_callable)
# For pair datasets we expand into multiple arms (one binary per arm).

def _ren_chrono_real(pair):     return renderers.render_chronometric_pair(pair, "real")
def _ren_chrono_ablated(pair):  return renderers.render_chronometric_pair(pair, "ablated")
def _ren_contra_a_amb(pair):    return renderers.render_contradiction_pair(pair, "mem_a_amb")
def _ren_contra_b_amb(pair):    return renderers.render_contradiction_pair(pair, "mem_b_amb")
def _ren_contra_a_exp(pair):    return renderers.render_contradiction_pair(pair, "mem_a_exp")
def _ren_contra_b_exp(pair):    return renderers.render_contradiction_pair(pair, "mem_b_exp")


JOBS: list[tuple[str, str, Callable[[dict[str, Any]], str]]] = [
    # Latent World — train
    ("data/latent_world/train_1k.jsonl",  "data/tokenized/latent_world/train_1k",  renderers.render_latent_world),
    ("data/latent_world/train_2k.jsonl",  "data/tokenized/latent_world/train_2k",  renderers.render_latent_world),
    ("data/latent_world/train_4k.jsonl",  "data/tokenized/latent_world/train_4k",  renderers.render_latent_world),
    ("data/latent_world/train_8k.jsonl",  "data/tokenized/latent_world/train_8k",  renderers.render_latent_world),
    # Latent World — test
    ("data/latent_world/test_16k.jsonl",  "data/tokenized/latent_world/test_16k",  renderers.render_latent_world),
    ("data/latent_world/test_32k.jsonl",  "data/tokenized/latent_world/test_32k",  renderers.render_latent_world),
    ("data/latent_world/test_64k.jsonl",  "data/tokenized/latent_world/test_64k",  renderers.render_latent_world),
    ("data/latent_world/test_128k.jsonl", "data/tokenized/latent_world/test_128k", renderers.render_latent_world),
    # Latent World — valid
    ("data/latent_world/valid_1k.jsonl",  "data/tokenized/latent_world/valid_1k",  renderers.render_latent_world),
    ("data/latent_world/valid_4k.jsonl",  "data/tokenized/latent_world/valid_4k",  renderers.render_latent_world),
    ("data/latent_world/valid_16k.jsonl", "data/tokenized/latent_world/valid_16k", renderers.render_latent_world),
    # Ambiguity
    ("data/ambiguity/train.jsonl",        "data/tokenized/ambiguity/train",        renderers.render_ambiguity),
    ("data/ambiguity/valid.jsonl",        "data/tokenized/ambiguity/valid",        renderers.render_ambiguity),
    # Consolidation
    ("data/consolidation/ladder_train.jsonl", "data/tokenized/consolidation/ladder_train", renderers.render_consolidation),
    # Chronometric pairs (split into real / ablated arms)
    ("data/chronometric_pairs/pairs.jsonl",  "data/tokenized/chronometric_pairs/pairs_real",     _ren_chrono_real),
    ("data/chronometric_pairs/pairs.jsonl",  "data/tokenized/chronometric_pairs/pairs_ablated",  _ren_chrono_ablated),
    # Contradiction pairs (4 arms)
    ("data/contradiction_pairs/pairs.jsonl", "data/tokenized/contradiction_pairs/mem_a_amb",     _ren_contra_a_amb),
    ("data/contradiction_pairs/pairs.jsonl", "data/tokenized/contradiction_pairs/mem_b_amb",     _ren_contra_b_amb),
    ("data/contradiction_pairs/pairs.jsonl", "data/tokenized/contradiction_pairs/mem_a_exp",     _ren_contra_a_exp),
    ("data/contradiction_pairs/pairs.jsonl", "data/tokenized/contradiction_pairs/mem_b_exp",     _ren_contra_b_exp),
    # Real text
    ("data/real_text/gutenberg.jsonl",       "data/tokenized/real_text/gutenberg",               renderers.render_real_text),
]


def tokenize_one(jsonl_path: Path, out_prefix: Path, render_fn: Callable[[dict[str, Any]], str], enc):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    n_lines = sum(1 for _ in open(jsonl_path))
    boundaries: list[tuple[int, int]] = []
    all_tokens: list[int] = []

    with open(jsonl_path) as f:
        for line in tqdm(f, total=n_lines, desc=out_prefix.name, leave=False):
            ex = json.loads(line)
            text = render_fn(ex)
            toks = enc.encode(text)
            start = len(all_tokens)
            all_tokens.extend(toks)
            boundaries.append((start, len(all_tokens)))

    # Flush
    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile(str(out_prefix) + ".tokens.bin")
    bounds = np.array(boundaries, dtype=np.int64)
    np.save(str(out_prefix) + ".boundaries.npy", bounds)

    meta = {
        "source_jsonl": str(jsonl_path),
        "n_examples": len(boundaries),
        "n_tokens": int(arr.size),
        "mean_tokens_per_example": float(arr.size) / max(1, len(boundaries)),
        "min_tokens": int((bounds[:, 1] - bounds[:, 0]).min()) if len(bounds) else 0,
        "max_tokens": int((bounds[:, 1] - bounds[:, 0]).max()) if len(bounds) else 0,
        "dtype": "uint16",
        "vocab": "gpt2",
    }
    with open(str(out_prefix) + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  {out_prefix.name}: {meta['n_examples']} examples, {meta['n_tokens']:,} tokens, "
          f"mean={meta['mean_tokens_per_example']:.0f}")
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=str, default=None, help="substring filter on output prefix")
    args = p.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    all_meta = {}
    for jsonl_path, out_prefix, render_fn in JOBS:
        if args.only and args.only not in out_prefix:
            continue
        if not Path(jsonl_path).exists():
            print(f"  skipped (missing): {jsonl_path}")
            continue
        meta = tokenize_one(Path(jsonl_path), Path(out_prefix), render_fn, enc)
        all_meta[out_prefix] = meta

    summary_path = Path("data/tokenized/SUMMARY.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_meta, f, indent=2)
    total_tok = sum(m["n_tokens"] for m in all_meta.values())
    total_ex = sum(m["n_examples"] for m in all_meta.values())
    print(f"\nTotal across {len(all_meta)} caches: {total_ex} examples, {total_tok:,} tokens")
    print(f"Summary at {summary_path}")


if __name__ == "__main__":
    main()
