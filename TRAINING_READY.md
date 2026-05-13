# IPCN — Training-Ready Status (v2)

Date: 2026-05-13
Repo: https://github.com/sam-siavoshian/AGI
Latest commit: `bbeaf79` (or later)

Everything below verified on CPU. Ready for NVIDIA DGX Spark transition.

---

## What's complete

### Model architecture (`model/`)

| Module | What it does |
|---|---|
| `config.py` | All hyperparameters + ablation flags locked in one dataclass |
| `chronometric.py` | Deterministic χ_t encoder (13-scale sinusoidal, 0 params) |
| `adapters.py` | LoRALinear with snapshot/restore for rollback |
| `memory.py` | 256-slot bank, top-1 hard slot assignment, Δτ-driven evolution |
| `pfc.py` | 2-layer Prefix-Forming Controller (4 heads, hidden 512) |
| `injection.py` | Route 2 broadcast preconditioner with λ_pre anneal |
| `core.py` | 8-layer decoder-only Transformer, vectorized causal+prefix mask, Route-3 LN modulation, split compute_hidden/decode for late retrieval |
| `late_retrieval.py` | A1 baseline path (memory consulted AFTER core) |
| `losses.py` | All 9 loss terms with log-space chronometric |
| `consolidation.py` | Teacher-student KL distillation |
| `ipcn.py` | Top-level forward_chunk supporting all 7 ablations |

**Total params: 102,086,794** (added ~800K for LateRetrievalHead since v1)

### Training pipeline (`model/`)

| Module | What it does |
|---|---|
| `dataset.py` | TokenizedCache (numpy memmap), SequentialChunkDataset, MixedDataset |
| `latent_world_loader.py` | Real per-chunk τ from event metadata |
| `replay_buffer.py` | Per-slot ring buffer, push on prefix-helpful chunks |
| `train.py` | Train step (6 of 9 losses) + real prefix attribution + replay push + consolidation pass with 2 validation gates |
| `checkpoint.py` | Save/load model + optimizer + RNG + cfg |
| `phases.py` | Phase scheduler with LoRA toggles |
| `run_phase.py` | Phase-aware CLI with `--use-real-tau`, `--enable-consolidation`, threshold overrides |
| `ablation_runner.py` | Train + eval A0..A6 variants |

### Eval harnesses (all 7 predictions)

| Pred | Module | Threshold | Untrained baseline |
|---|---|---|---|
| H1 D_0 | `predictions.py::H1_synthetic_check` | > 0.1 | 0.519 PASS |
| H2 probe | `h2_probe.py` | ≥ 0.80 | 0.50-0.75 |
| H3 ordering | `ablation_runner.py` | Acc(A3) − Acc(A1) ≥ 0.03 | A0=A1=0, A2+>0.4 (D_0) |
| H4 CTI | `h4_cti.py` | > 0.7 | needs pre/post ckpts |
| H5 silent-gap | `h5_evolution.py` | ≥ 0.15 | 0.000 |
| H6 chronometric | `predictions.py::H6_pairs_check` | KL ≥ 0.10 | ≈ 0 |
| H7 contradiction | `h7_contradiction.py` | KL_amb ≥ 0.5, KL_exp ≤ 0.1 | 6e-4 / 6e-4 |

Unified runner: `model/eval_all.py` → `reports/<name>.md` + JSON.

### Consolidation safety gates

1. LM drift gate: KL(pre_logits || post_logits) ≤ cfg.kl_drift_threshold (0.02)
2. Held-out accuracy gate: pre_accuracy − post_accuracy ≤ cfg.eps_drop (0.02)
3. (Spec also calls for contradiction-floor gate; implemented as held-out accuracy proxy)

On gate fail: snapshot restored, `committed=False` returned, training log records `ROLLBACK` with reason.

### Datasets (verified)

- Latent World: 49.8K streams, 11 bins, real τ extractable
- Ambiguity Suite: 110K examples, 5 families × 2 senses
- Consolidation Ladder: 20K rules
- Chronometric pairs: 10K (real Δτ + ablated arms)
- Contradiction pairs: 5K (4 memory × input arms)
- Gutenberg: 6,348 chunks (1.4M tokens)
- Tokenized binary caches: 21 files, 226K examples, 323M tokens

---

## Verified on CPU

| Test | Result |
|---|---|
| Smoke test (all losses, 2 chunks, backward) | PASS |
| 100-step Phase 0 LM descent | 11.0 → 1.23 (ppl 58K → 3.4) |
| Phase 0 → Phase 1 transition (resume + base freeze) | 100M → 114K trainable |
| Consolidation pipeline (triggers + logging + gates) | runs cleanly |
| Ablation runner (A0/A1/A2/A3 × 20 steps) | D_0 ordering correct |
| All 7 prediction harnesses on untrained | shape-correct, values sensible |

---

## Phase 0 launch sequence on DGX Spark

```bash
# Step 1: sync repo
rsync -avz --exclude='.venv' --exclude='logs' --exclude='checkpoints' \
  "/Users/samsiavoshian/Desktop/Coding Stuff/Time-Model/" \
  root1@100.83.86.5:Desktop/Time-Model/

# Step 2: sync tokenized data
rsync -avz "/Users/samsiavoshian/Desktop/Coding Stuff/Time-Model/data/tokenized/" \
  root1@100.83.86.5:Desktop/Time-Model/data/tokenized/

# Also sync raw Latent World JSONL for --use-real-tau
rsync -avz "/Users/samsiavoshian/Desktop/Coding Stuff/Time-Model/data/latent_world/" \
  root1@100.83.86.5:Desktop/Time-Model/data/latent_world/

# Step 3: deps
ssh root1@100.83.86.5 'cd Desktop/Time-Model && export PATH="$HOME/.local/bin:$PATH" && uv sync'

# Step 4: smoke Phase 0 (1k steps, ~10-20 min on GPU)
ssh root1@100.83.86.5 'cd Desktop/Time-Model && \
  export PATH="$HOME/.local/bin:$PATH" && \
  nohup caffeinate -dimsu uv run python -m model.run_phase \
    --phase 0 --steps 1000 --device cuda --use-real-tau \
    --out-ckpt checkpoints/phase0_smoke.pt \
    > logs/phase0_smoke.log 2>&1 & disown'

# Step 5: eval Phase 0 smoke
ssh root1@100.83.86.5 'cd Desktop/Time-Model && \
  export PATH="$HOME/.local/bin:$PATH" && \
  uv run python -m model.eval_all \
    --checkpoint checkpoints/phase0_smoke.pt \
    --out reports/phase0_smoke.md \
    --n-trials 50 --device cuda'

# Step 6: if H1/H6 trend positive, full Phase 0 (100k steps)
# Step 7: chain Phase 1 with --resume + --enable-consolidation
# Step 8: chain Phase 2 + 3
# Step 9: run ablation_runner across A0..A6 for H3
```

---

## What's deferred (post-Spark, not blockers)

- Multi-stream batching (current batch=1, GPU underutilized but trainable)
- wandb / TensorBoard logging (JSON logs sufficient for v1)
- Contradiction-floor gate using real contradiction_pairs in-loop (currently approximated by held-out accuracy gate)
- Long-stream test eval optimization (test_16k-128k bins)
- Auto-tuning of t_stable / tau_cons based on observed kappa distributions

---

## All 25 tracked tasks complete

All eval harnesses operational. All 7 predictions have pass criteria + untrained baselines committed. Architecture frozen. Pre-registration locked.

Standing by for Spark green light.
