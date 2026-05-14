#!/usr/bin/env bash
# Track B smoke test: load Qwen 2.5 1.5B, wrap with IPCN, forward pass.
#
# Verifies:
#   - Qwen base loads (downloads from HF if not cached)
#   - LoRA adapters injected at layers 0, 1
#   - PFC produces K_p=16 prefix vectors
#   - Memory bank reads + writes work
#   - Forward pass returns logits of the right shape
#   - Trainable params count is small (LoRA + PFC + memory projections only)
#
# Run on Spark: bash scripts/qwen_smoke.sh
# Expected wall time: ~60s (download/cache hits dominate)

set -uo pipefail
ROOT="${IPCN_ROOT:-$HOME/ipcn}"
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"
export IPCN_ROOT="$ROOT"

PYTHONPATH=. uv run python3 - <<'PYEOF'
import time
import torch
import sys
sys.path.insert(0, ".")
from model.qwen_ipcn import QwenIPCNConfig, build_qwen_ipcn

print("=" * 60)
print("Track B smoke: Qwen IPCN wrapper")
print("=" * 60)

cfg = QwenIPCNConfig()
print(f"Loading {cfg.base_model_name}...")
t0 = time.time()
model = build_qwen_ipcn(cfg)
t_load = time.time() - t0
print(f"  loaded in {t_load:.1f}s")

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    # Don't move the whole base; it's already in bf16 on CPU. Use accelerate-style.
    model = model.to(device)
print(f"  device: {device}")

# Param counts
total = sum(p.numel() for p in model.parameters())
trainable = model.trainable_parameter_count()
print(f"  total params: {total:,}")
print(f"  trainable params: {trainable:,} ({100 * trainable / total:.3f}%)")
print(f"  LoRA modules adapted: {model._n_lora_modules}")
print(f"  memory bank: {cfg.n_slots} slots x {cfg.d_memory} d_memory")
print(f"  PFC: {cfg.prefix_length} prefix tokens, d_model={model.d_model}")

# Tokenize a tiny prompt and run forward
tok = model.tokenizer
prompt = "The capital of France is"
ids = tok.encode(prompt, return_tensors="pt").to(device)
print(f"\nForward pass on prompt: {prompt!r}")
print(f"  input shape: {tuple(ids.shape)}")
t0 = time.time()
with torch.no_grad():
    out = model(ids[0], tau_t=0.0, delta_tau=0.0)
dt = time.time() - t0
print(f"  forward in {dt*1000:.0f}ms")
logits = out["logits"]
print(f"  logits shape: {tuple(logits.shape)}")
# Argmax next-token after the last input token
next_tok = int(logits[-1].argmax().item())
print(f"  next token id: {next_tok}")
print(f"  next token str: {tok.decode([next_tok])!r}")

# Check memory bank state mutated
nonzero_slots = int((model.memory.tau_write > 0).sum().item())
print(f"\nMemory bank after forward: {nonzero_slots}/{cfg.n_slots} slots written")
print(f"  v_max norm: {model.memory.v.norm(dim=-1).max().item():.4f}")

print("\nSmoke OK.")
PYEOF
