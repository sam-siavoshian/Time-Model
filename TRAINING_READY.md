# IPCN — Training-Ready Status

Date: 2026-05-12
Commit: latest on `main`

Everything below is verified on CPU. Ready to move to NVIDIA DGX Spark for Phase 0 training when you give the go.

---

## Verified end-to-end

### Architecture (model/)

- `config.py` — all hyperparameters locked, single source of truth
- `chronometric.py` — deterministic χ_t encoder (13-scale sinusoidal, zero learned params)
- `adapters.py` — LoRALinear with snapshot/restore for rollback
- `memory.py` — 256-slot bank with random unit-norm key init, top-1 hard slot assignment, Δτ-driven evolution
- `pfc.py` — 2-layer Prefix-Forming Controller (4 heads, hidden 512)
- `injection.py` — Route 2 broadcast preconditioner with λ_pre anneal schedule
- `core.py` — 8-layer decoder-only Transformer, vectorized prefix+causal mask, Route-3 LayerNorm modulation on layers 1-2
- `losses.py` — all 9 loss terms implemented
- `consolidation.py` — teacher-student KL distillation + 3 validation gates + mechanical rollback
- `ipcn.py` — top-level `forward_chunk` matching SPEC.tex inference algorithm

**Total params: 101,298,826** (target was 50-150M)

### Training infrastructure (model/)

- `dataset.py` — TokenizedCache (numpy memmap), SequentialChunkDataset (per-example chunk iterator with cross-chunk memory), MixedDataset (weighted multi-cache mix)
- `train.py` — full training step with 6 of 9 losses + real prefix attribution sampling on 15% of chunks
- `checkpoint.py` — save/load model + optimizer + cfg + RNG + train_step
- `phases.py` — phase scheduler with per-phase LoRA toggles
- `run_phase.py` — single-command phase runner with checkpoint passing

### Evaluation harnesses (model/)

- `predictions.py` — H1 (D_0 layer-0 memory-swap), H6 (chronometric Δτ ablation)
- `h7_contradiction.py` — H7 (memory-swap + explicit-input KL constraint)

### Datasets (data/)

- Latent World: 49,800 streams across 11 splits (1k-128k token bins), 1.4 GB JSONL
- Memory-Biased Ambiguity Suite: 110,000 examples, 70 MB
- Consolidation Ladder: 20,000 rules
- Chronometric pairs: 10,000 with real + ablated Δτ arms
- Contradiction pairs: 5,000 with 4 arms each
- Gutenberg corpus: 6,348 chunks, 1.4M tokens
- **All tokenized to 21 binary caches: 226,148 examples, 323M tokens, 620 MB**
- All schema-verified by `data_gen/verify_data.py`. Tokenized roundtrip decode is byte-exact.

---

## Smoke test results (CPU)

| Test | Result |
|---|---|
| Param count | 101,298,826 |
| Forward chunk time | 0.235s on CPU (was 0.78s before mask vectorization) |
| Cross-chunk alpha diff | 1.36e-4 (memory state affects PFC attention) |
| Slots touched per write | 15/256 (top-1 hard assignment, K_w=16) |
| LM loss baseline | ~11 on random targets (≈ log(50257) = 10.82) |
| 100-step training LM | 11.0 → 1.23 (perplexity 58K → 3.4) |
| All 9 loss terms | non-NaN |
| Backward pass | 201/272 param tensors receive grads |

## Pre-training sanity predictions

| Hypothesis | Untrained model | Threshold | Status |
|---|---|---|---|
| H1 D_0 (memory-swap) | 0.519 ± 0.018 | > 0.1 | passes (architecturally) |
| H6 KL (chronometric ablation) | 1e-4 | nonzero | passes (chi reaches output) |
| H7 KL_amb / KL_exp | 6e-4 / 6e-4 | > 0.5 / < 0.1 | passes only KL_exp |

H1 architectural signal is strong even untrained. H6 + H7 trends positive but small. Both expected to grow with Phase 0 training.

---

## Phase parameter budgets (verified)

| Phase | Trainable params | What's on |
|---|---|---|
| 0 sanity | 100,962,954 (99.67%) | full base, LoRA frozen |
| 1 PFC consolidation | 114,688 (0.11%) | base frozen, PFC LoRA only |
| 2 early-core | 335,872 (0.33%) | base frozen, PFC + core layers 0-2 LoRA |
| 3 mixed LM | 335,872 (0.33%) | same as Phase 2 |

---

## Spark transition plan

When ready to train, do:

1. **Sync repo to DGX Spark:**
   ```bash
   rsync -avz --exclude='.venv' --exclude='logs' \
     "/Users/samsiavoshian/Desktop/Coding Stuff/Time-Model/" \
     root1@100.83.86.5:Desktop/Time-Model/
   ```

2. **Sync tokenized data (~620 MB):**
   ```bash
   rsync -avz "/Users/samsiavoshian/Desktop/Coding Stuff/Time-Model/data/tokenized/" \
     root1@100.83.86.5:Desktop/Time-Model/data/tokenized/
   ```

3. **Install deps on Spark:**
   ```bash
   ssh root1@100.83.86.5 \
     'cd Desktop/Time-Model && export PATH="$HOME/.local/bin:$PATH" && uv sync'
   ```

4. **Phase 0 sanity run (1k steps first, then full 100k):**
   ```bash
   ssh root1@100.83.86.5 \
     'cd Desktop/Time-Model && export PATH="$HOME/.local/bin:$PATH" && \
      nohup caffeinate -dimsu uv run python -m model.run_phase \
        --phase 0 --steps 1000 --device cuda --out-ckpt checkpoints/phase0_smoke.pt \
        > logs/phase0_smoke.log 2>&1 & disown'
   ```

5. **Eval predictions on Phase 0 checkpoint:**
   ```bash
   uv run python -m model.predictions --trials 100
   uv run python -m model.h7_contradiction --checkpoint checkpoints/phase0_smoke.pt --n 50
   ```

6. **If H1 D_0 > 0.1 and LM perplexity descending:** scale to full 100k Phase 0.

7. **Phase 1-3:** chain via `--resume`.

---

## Still left to build (post-Spark)

- **H2 linear probe** (Task #24): train sklearn logistic regression on H^0 / H^1, predict memory-conditioned sense
- **H3 ablation matrix** (Task #25): wire ablation flags to disable {slots, prefix, broadcast, evolution, consolidation} independently. Train A0-A6 at identical compute.
- **Real-Δτ extraction**: parse "At minute XXXXX" from Latent World stream text, pass to forward_chunk
- **Multi-stream batching**: currently batch=1 (one stream at a time). For full GPU throughput, support batched memory banks.
- **Better logging**: wandb or TensorBoard integration
- **Consolidation pass integration in train loop**: currently consolidation.py has the algorithm; train_step doesn't trigger it. Wire in for Phase 1+.

All deferred — none blocks Phase 0.

---

## Risk register

- Phase 0 fails to drive LM loss down → likely PFC interferes too aggressively with content. Mitigation: lower λ_pre_init or disable Route-3 LN modulation as ablation.
- Memory bank slot collapse (few slots dominate writes) → slot_util_loss should prevent; if not, lower η_sim or add stronger diversity loss weight.
- H1 D_0 drops below 0.1 after training → prefix has been suppressed by precision loss too aggressively. Cap w_precision.
- DGX Spark CUDA out-of-memory on chunk_length=256, batch=1 → unlikely at 101M params, but fall back to width=384 if needed.
- Per-step time too slow on CPU for any meaningful Phase 0 run → confirmed. CPU is only for smoke tests; Spark required for full training.

---

## Bottom line

**Code: ready. Data: ready. Architecture: locked. Pre-registration: locked.**

Phase 0 training can run as soon as we move to DGX Spark.

When you call it, the sequence is: sync → smoke 1k steps → eval predictions → if green, full Phase 0 (100k steps).
