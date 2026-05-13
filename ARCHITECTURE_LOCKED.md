# IPCN Architecture — LOCKED

Date locked: 2026-05-12
Canonical spec: `SPEC.tex` (authoritative; this file is the index).

This document pins every architectural decision for v1. No more debate, no more
exploration. Implementation follows.

---

## Lock list

### 1. Topology

- **Core network:** decoder-only Transformer, 8 layers, width 512, 8 heads, FFN 2048
- **Prefix-forming controller (PFC):** 2-layer Transformer over prefix tokens, 4 heads, hidden 512, FFN 1024, LoRA rank 8
- **Episodic memory bank:** 256 slots, slot dimension 256
- **Temporal self-state z:** 128-dim GRU-updated vibe vector
- **LoRA adapters Ω:** rank 8 (start), 16 (scale-up). Live in PFC + core layers 0-2 only.
- **Chronometric encoder χ_t:** deterministic, 13 timescales 𝒯 = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536}

### 2. Slot record

Each of 256 slots holds: key k, value v, dynamics state q, age a, usage u, confidence c, plasticity ρ, conflict δ. Plus temporal metadata: τ_write, τ_use, χ_slot (learned temporal signature).

### 3. Three injection routes

1. **MANDATORY:** prefix prepending H = [P; E(X)] with full attention from content tokens to prefix
2. **STRONGLY RECOMMENDED:** broadcast preconditioning. Tokens preconditioned via gated residual `ẽ = LN(e + λ_pre · γ ⊙ W_b · b)`. λ_pre = 0.5 → 1.0 over steps 0-100k.
3. **OPTIONAL (layers 1-2 only):** LayerNorm modulation via FiLM-style Γ, B from prefix broadcast. α_film = 0.1.

### 4. Loss objective (9 terms)

```
ℒ = ℒ_LM
  + 0.02·ℒ_pre-influence
  + 0.02·ℒ_precision
  + 0.05·ℒ_mem-predict
  + 0.001·ℒ_diversity
  + 0.001·ℒ_slot-util
  + 0.02·ℒ_evolution
  + 0.03·ℒ_chrono
  + 0.1·ℒ_cons    (during consolidation phases only)
```

### 5. Consolidation rule

A slot enters eligibility when:
- usage score κ = log(1+u)·c·(1-δ)·ReLU(ū^prefix)·(1-ρ) > 3.0
- AND slot stable ≥ 512 chunks

Then run teacher-student KL distillation into LoRA, validate (held-out accuracy drop < 2%, contradiction drop < 1%, LM KL drift < 0.02), commit or rollback.

### 6. Training schedule

| Phase | Step count | What's on |
|---|---|---|
| 0 — sanity | 50-100k | episodic memory + prefix injection + evolution. Ω frozen. |
| 1 — PFC consolidation | 50k | LoRA adapters in PFC only |
| 2 — early-core consolidation | 50k | LoRA in layers 1-2 added |
| 3 — mixed LM | 100k | synthetic + real text. Watch perplexity drift. |

### 7. Hyperparameters (v1 prototype)

| Item | Value |
|---|---|
| Training precision | bf16 |
| Optimizer | AdamW |
| Base learning rate | 3e-4 |
| LoRA consolidation LR | 1e-5 to 5e-5 |
| Chunk length L | 256 tokens |
| Local content window | 1024 tokens |
| Prefix length K_p | 32 tokens |
| Backprop through chunks | 4, then detach memory values |
| Consolidation frequency | every 256 chunks after warmup |
| GPU target | DGX Spark (Blackwell, 128 GB unified) primary, Mac mini M4 for probes |

### 8. Hard pass criteria (Phase 0 first experiment)

| Metric | Threshold |
|---|---|
| Acc(A3) − Acc(A1) on ambiguity | ≥ 0.10 |
| I_pre(A3) when prefix is useful | ≥ 0.25 |
| Acc_explicit(A3) − Acc_explicit(A1) on contradiction | ≥ −0.02 |
| CTI(A5) after consolidation | ≥ 0.70 |
| OLP(A5) / OLP(A3) drift | ≤ 1.05 |

### 9. Ablation matrix (locked)

| Model | Slots | Prefix | Broadcast | Evolution | Consolidation |
|---|---|---|---|---|---|
| A0 | no | no | no | no | no |
| A1 | yes | late read only | no | no | no |
| A2 | yes | yes | no | no | no |
| A3 | yes | yes | yes | no | no |
| A4 | yes | yes | yes | yes | no |
| A5 | yes | yes | yes | yes | PFC only |
| A6 | yes | yes | yes | yes | PFC + layers 1-2 |

### 10. Seven falsifiable predictions

1. D_0 > 0.1 on memory-swap (memory must change layer 0)
2. Layer 0/1 probes decode memory-conditioned sense ≥ 80% accuracy
3. Ablation order: A0 < A1 < A2 < A3 < A4 < A5 < A6 on ambiguity; weakened if Acc(A5) − Acc(A1) < 0.03
4. CTI > 0.7 with contradiction accuracy drop < 1%
5. Acc(evolve) − Acc(static) ≥ 0.15 at 64k context + 512 silent minutes
6. Acc(Δτ-aware) − Acc(Δτ-ablated) ≥ 0.10 on duration-sensitive; KL ≤ 0.1 on duration-insensitive
7. KL(amb) ≥ 0.5 AND KL(explicit) ≤ 0.1 on memory swap

---

## What's NOT locked (future work)

- Vocabulary: GPT-2 BPE for v1. May swap to Qwen3 tokenizer if grafting on pretrained base.
- Build path: from-scratch 50-150M params for v1 prototype. Grafting on pretrained base deferred.
- Multilingual: English-only for v1.
- Scale: ~100M params is the v1 target. Scale-up after Phase 0 pass criteria green.

## Implementation order (next tasks)

1. **Task 8** — Draft OSF pre-registration document. Lock seven predictions with thresholds + multiverse outcome narratives BEFORE any training.
2. **Task 9** — Code minimal IPCN prototype in PyTorch:
   - `model/core.py`: decoder-only Transformer (8 layers, width 512)
   - `model/pfc.py`: prefix-forming controller (2-layer Transformer over slots)
   - `model/memory.py`: 256-slot bank with key/value/metadata + evolution dynamics
   - `model/adapters.py`: LoRA stack (rank 8) on PFC + layers 0-2
   - `model/chronometric.py`: deterministic χ_t encoder
   - `model/injection.py`: three injection routes (prepend, broadcast, LayerNorm modulation)
   - `model/losses.py`: 9-term objective
   - `model/consolidation.py`: teacher-student distillation + validation gates + rollback
   - `model/ipcn.py`: top-level forward_chunk loop matching SPEC.tex inference algorithm
3. **Phase 0 sanity training** on DGX Spark.

Spec is locked. No more architecture decisions. Build now.
