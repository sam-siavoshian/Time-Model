# IPCN Pre-Registration

**Project:** Involuntary Prefix Consolidation Networks (IPCN)
**Author:** Saam Siavoshian
**Date locked:** 2026-05-12
**Canonical architecture:** `SPEC.tex` / `ARCHITECTURE_LOCKED.md`
**Repository:** https://github.com/sam-siavoshian/AGI

This document locks every hypothesis, threshold, decision rule, and outcome narrative BEFORE any training run. Once registered (intended OSF upload), no changes without an amendment record.

---

## 1. Research question

Can a transformer architecture combining (i) pre-computational memory injection, (ii) real elapsed wall-clock time as a causal substrate, and (iii) usage-driven LoRA consolidation outperform retrieval-augmented and KV-cache baselines on memory-biased ambiguity, silent-gap temporal dynamics, and rule consolidation tasks, while remaining suppressible by explicit contradictory evidence?

## 2. Hypotheses (the 7 falsifiable predictions)

Each hypothesis has a numerical threshold. Each carries a pre-registered fail narrative (Section 5).

**H1 — Memory swap changes layer 0.**
For identical input X with memory states M^A and M^B:
```
D_0 = ||H_0^A − H_0^B||_F / (||E(X)||_F + ε)
```
**Predict:** D_0 > 0.1 on memory-biased ambiguity inputs.
**Reject if:** D_0 ≤ 0.1 (late-retrieval baseline level).

**H2 — Early probes decode memory-conditioned sense.**
Train linear probes on H^0 and H^1 hidden states for ambiguity inputs (e.g., "He saw the bat near the entrance" under baseball vs cave memory).
**Predict:** probe accuracy ≥ 80% on layer-0 or layer-1.
**Reject if:** probe accuracy < 80% (memory not in early representations).

**H3 — Ablation order: B5 > B4 > B3 > B2 > B1 > B0.**
Full IPCN (consolidation off) beats prefix+broadcast > prefix-only > mid-layer injection > late retrieval > no memory.
**Predict:** ordering holds AND Acc(B5) − Acc(B1) ≥ 0.03 on Memory-Biased Ambiguity Suite.
**Reject (weak) if:** Acc(B5) − Acc(B1) < 0.03.

**H4 — Consolidation Transfer Index (CTI) ≥ 0.7.**
For an episodic slot m_i used heavily:
```
CTI_i = (Acc_post^without − Acc_pre^without) / (Acc_pre^with − Acc_pre^without + ε)
```
**Predict:** CTI > 0.7 AND contradiction accuracy drop < 1%.
**Reject if:** CTI ≤ 0.7 OR contradiction accuracy drop ≥ 1%.

**H5 — Silent-gap memory evolution.**
On Temporal Latent World streams with 64k context + 512 silent minutes:
**Predict:** Acc(IPCN_evolve) − Acc(IPCN_static) ≥ 0.15.
**Reject if:** gap < 0.15 (evolution mechanism unnecessary).

**H6 — Chronometric substrate.**
On paired Δτ-real vs Δτ-ablated streams:
```
duration-sensitive tasks:    Acc(Δτ-aware) − Acc(Δτ-ablated) ≥ 0.10
duration-insensitive tasks:  KL(p(y|X, Δτ_a) || p(y|X, Δτ_b)) ≤ 0.1
```
**Predict:** both conditions hold.
**Reject if:** either fails (time substrate is fake).

**H7 — Explicit evidence overrides memory.**
On ambiguous vs explicit input, with memory states M^A vs M^B:
```
KL(p(y|X_amb, M^A) || p(y|X_amb, M^B)) ≥ 0.5
KL(p(y|X_explicit, M^A) || p(y|X_explicit, M^B)) ≤ 0.1
```
**Predict:** both hold.
**Reject if:** either fails (model is overconsolidating).

## 3. Methods (locked)

### Architecture
Per `SPEC.tex` and `ARCHITECTURE_LOCKED.md`. No deviation without amendment.

Core: decoder-only Transformer, 8 layers, width 512, 8 heads, FFN 2048.
PFC: 2-layer Transformer over 32 prefix tokens, 4 heads, hidden 512, FFN 1024.
Memory bank: 256 slots × 256 dim, full metadata (k, v, q, a, u, c, ρ, δ + τ_write, τ_use, χ_slot).
Chronometric encoder: 13-scale sinusoidal, 𝒯 = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536}.
LoRA adapters: rank 8 (start), 16 (scale-up), in PFC + core layers 0-2 only.
Three injection routes: prepend (mandatory) + broadcast (recommended) + LayerNorm modulation (optional, layers 1-2).

### Training schedule

| Phase | Steps | Adapters | Goal |
|---|---|---|---|
| 0 | 50-100k | frozen | sanity: prefix affects layer 0, A3 > A1 on ambiguity |
| 1 | 50k | PFC only | warmup consolidation, validate < 2% acc drop on slot removal |
| 2 | 50k | PFC + layers 1-2 | early-core consolidation |
| 3 | 100k | all of above | mixed LM, watch perplexity drift |

Total: ~250k steps. Compute: NVIDIA DGX Spark + Mac mini M4. Zero rented cloud.

### Data
Per `data/README.md` and `data/VERIFICATION.md`. Verified clean.

- Temporal Latent World: 45k train (1k/2k/4k/8k bins) + 3.7k test (16k/32k/64k/128k) + 1.1k valid
- Memory-Biased Ambiguity Suite: 100k train + 10k valid
- Consolidation Ladder: 20k rules
- Chronometric pairs: 10k (real + ablated arms)
- Contradiction pairs: 5k (4 arms each)
- Project Gutenberg corpus: 10 books, ~6,348 chunks, 1.4M tokens
- Total: 226,148 examples, 323M tokens, all schema-verified

### Ablation matrix (locked)

| Model | Slots | Prefix | Broadcast | Evolution | Consolidation |
|---|---|---|---|---|---|
| A0 | no | no | no | no | no |
| A1 | yes | late read only | no | no | no |
| A2 | yes | yes | no | no | no |
| A3 | yes | yes | yes | no | no |
| A4 | yes | yes | yes | yes | no |
| A5 | yes | yes | yes | yes | PFC only |
| A6 | yes | yes | yes | yes | PFC + layers 1-2 |

All ablations trained at identical parameter budget on identical data.

### Statistical analysis plan

- Each hypothesis tested per-example, then aggregated to a single accuracy/KL/D_0 number per ablation.
- Confidence intervals: bootstrap 1000 iterations, 95% percentile interval.
- Multiple-comparisons correction across the 7 hypotheses: Bonferroni at family-wise α=0.05 (per-hypothesis α=0.0071).
- For ordering hypothesis (H3): rank correlation Kendall's τ between predicted and observed ablation accuracy, expect τ > 0.7.
- For continuous-threshold hypotheses (D_0, CTI, KL gaps): point estimate vs threshold. Pre-register exact estimator.

### Computational reproducibility

- All synthetic data deterministic by seed (see `data/README.md` for seed table).
- Model code committed to https://github.com/sam-siavoshian/AGI before training.
- Training run logs + checkpoints + seeds committed at experiment end.
- Each ablation A0-A6 trained from 3 random seeds; report mean ± std.

## 4. Decision rules

For each hypothesis H_i:
- **PASS:** point estimate beats threshold AND 95% bootstrap CI lower bound also beats threshold.
- **FAIL:** point estimate misses threshold OR 95% CI lower bound misses.
- **AMBIGUOUS:** point passes but CI lower bound fails. Report as "trending positive, not significant."

Overall outcome:
- **CLEAN PASS:** 6 or 7 of 7 hypotheses PASS.
- **MIXED:** 3-5 hypotheses PASS.
- **FAILED CORE:** 0-2 hypotheses PASS.

## 5. Pre-registered outcome narratives (the multiverse)

We commit to one of these four narratives before training. The paper writes itself once results land.

### A. CLEAN PASS narrative
"IPCN's pre-computational memory + chronometric substrate + LoRA consolidation passes 6 or 7 of 7 falsifiable predictions. The architecture is the first to combine these three primitives and pass under pre-registered thresholds. We claim a new memory-architecture topology for time-aware persistent learning. Releases: code, dataset, model weights, leaderboard."

### B. MIXED narrative (3-5 pass)
"IPCN passes some predictions, fails others. We document which predictions hold and which break. The architecture is partially supported. Per-failing-prediction analysis: which mechanism is responsible? Either: (a) consolidation works but layer-0 effect doesn't, suggesting prefix injection point should move, or (b) layer-0 effect holds but consolidation fails, suggesting LoRA capacity is wrong. Concrete revision proposed."

### C. FAILED CORE narrative (0-2 pass)
"IPCN does not pass its own pre-registered tests. We report negative results in detail. Late retrieval matches pre-computational injection; consolidation breaks more than it teaches; time substrate is no better than token counts. Lessons: which assumption was wrong? Where would future work need to start over? This is a publishable null result because the pre-registration is on file."

### D. UNCLEAR / ABORTED narrative
"Training instability or compute exhaustion prevents completing all phases. We report what was reached, what wasn't, and why. No claims of success or failure on hypotheses that didn't reach evaluation. Full transparency on training run logs."

## 6. Stopping rules + safety constraints

- Training stops if Phase 3 LM perplexity exceeds 1.05 × baseline LM perplexity.
- Consolidation rollback if contradiction accuracy drops > 1% or LM KL drift > 0.02.
- No hypothesis is added or removed post-hoc.
- No threshold is moved post-hoc.
- Any deviation requires a dated AMENDMENT log entry in this file.

## 7. Hardware + compute

- NVIDIA DGX Spark: primary training (128 GB unified Blackwell, fp4-capable)
- Mac mini M4: data prep, eval, small probes, validation runs
- Zero rented cloud for v1
- Estimated wall-clock: 4-6 months solo

## 8. Timeline

| Phase | Target completion |
|---|---|
| Pre-reg lock | 2026-05-12 (this document) |
| Prototype code complete | 2026-05-26 (Task #9) |
| Phase 0 training | 2026-06-09 |
| Phase 1-2 training | 2026-07-07 |
| Phase 3 training | 2026-08-04 |
| Eval + analysis | 2026-08-18 |
| First paper draft | 2026-09-15 |

## 9. Transparency commitments

- All training run logs committed to repo at experiment end (or at hypothesis evaluation point, whichever comes first).
- All seeds documented.
- All hyperparameter changes from this document recorded as amendments.
- If H1-H7 fail, narrative C is written and posted regardless of outcome.

## 10. Amendments

(None as of 2026-05-12. Any future deviation logs here with date.)

---

**This pre-registration is the contract.** Code + data + thresholds + narratives are locked. Run the experiment honestly. Report whatever comes back.
