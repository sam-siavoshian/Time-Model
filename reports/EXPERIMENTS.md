# IPCN experiments log (2026-05-13 to 2026-05-15)

Consolidated record of every training run, eval, and result.

## Track A: 102M from-scratch IPCN

Built on 102M-param decoder-only transformer trained from random init on
synthetic Latent World + Consolidation Ladder data. GPT-2 tokenizer.

| Phase | Steps | Final LM | CTI | Notes |
|---|---|---|---|---|
| Phase 0 sanity | 50000 | 0.60 | n/a | Baseline LM training. All H1-H7 near zero (expected) |
| Phase 1 v2 | 30000 | 0.10 | 0 | All 117 consolidation passes skipped (kappa<=tau_cons) |
| Phase 1 v4 | 20000 | 0.10 | 0 | tau_cons override -> 43 commits, 6 rollbacks. Eval CTI=0 (all 34 rules skipped at slot effect<5pp) |

**Track A verdict:** mechanism (consolidation gates, KL drift, rollback)
executes safely. Memory ablation does not affect accuracy on any test
rule because the 102M model never learned the rule task at all. No
behavioral pathway from memory to output.

Artifacts:
- `reports/phase0_baseline.md` — pre-consolidation eval
- `reports/alerts_ipcn_phase1_v4.jsonl` — full alert history (43 commits)
- `logs/phase0_ipcn_phase0_initial.jsonl` — Phase 0 training log
- `logs/phase1_ipcn_phase1_v4.jsonl` — Phase 1 with consolidation events
- `checkpoints/phase0_ipcn_phase0_initial_final.pt` (Spark, 1.2 GB)
- `checkpoints/phase1_ipcn_phase1_v4_final.pt` (Spark, 392 MB)

## Track B: Qwen 2.5 1.5B-Instruct + IPCN wrapper

Frozen Qwen base + IPCN scaffolding (PFC + memory bank + LoRA). Trained
on memorize-recall conversations. Eval: feed fact via memory bank, ask
recall question, compare with/without/shuffled memory.

| Run | Arch | Data | Loss | Recall delta (with-without) |
|---|---|---|---|---|
| v1 | prefix layer 0, LoRA layers 0-1, full LM loss | pool, short convs | n/a | -- (template guessing) |
| v2 | prefix layer 0, LoRA layers 0-1, answer-mask | pool, all in 1 chunk | 1e-7 | **0.000** (trivial in-context) |
| v3b | prefix layer 0, LoRA layers 0-1, longer convs | pool | 0.6-0.8 | **0.000** |
| v4 | prefix layer 0, LoRA all 28 layers + rank 16 | pool | 0.2-1.2 | **-0.02** |
| v5 | cross-attn layers 4/12/20 + Identity-V | nonce + 20% negatives | n/a | **0.000** (collapse to "I do not know") |
| v6 | cross-attn 4/12/20 + Identity-V, gate=0.5 init | nonce, no negatives | n/a | **0.000** (confabulate Qwen priors) |
| v7 | cross-attn 4/12/20 + Identity-V, gate=0.5 | single-token letters {A..H} | 0.7 | **0.000** (strict eval) |

**Track B verdict:** memory routing does NOT work on a frozen Qwen base
across any of seven architecture+data combinations. The mechanism
EXECUTES (memory writes, cross-attn injects, gate opens) but the
specific-value signal does not survive the residual stream from
injection layer to lm_head.

Literature confirms: Petrov & Liang (arxiv 2310.19698) prove prefix-
tuning at layer 0 cannot redirect attention; benchmark of 6 frozen-base
memory architectures (arxiv 2603.16413) puts prefix at 0.02% recall on
Flan-T5 vs cross-attention at 11.91%. v5-v7 attempted the cross-
attention pattern but still hit zero.

Artifacts:
- `reports/qwen_recall_v{1-7,7b}.json` — three-condition recall results
- `logs/qwen_ipcn_v{1-7}.jsonl` — training logs
- `checkpoints/qwen_ipcn_v{1-7}.pt` (Spark only, 62-82 MB each)

## Cumulative bug fixes from rollout (13 fixes)

See PAPER.md section 21.4. Production tooling shipped: atomic checkpoint
save, named-param optimizer-state restore across phase transitions,
crash-vs-completion sentinel, safety stall + start-stall watchdogs, CTI
denominator filter, fresh-slot conflict false positive, stale pad mask,
replay buffer pollution, monitor blindspots, preflight ROOT portability,
launcher pipefail breakage, single-chunk explosion false-fire, RNG
cross-device load. All shipped via commits `d48de1a` through `9d8216a`.

## Falsified hypotheses

1. **Prefix-prepend at Qwen layer 0 routes value-level memory.** False.
   v1-v4 across multiple data designs, multiple LoRA breadths.
2. **Cross-attention injection at Qwen layers 4/12/20 routes value-level
   memory under Identity-V on frozen base.** False. v5-v7 across nonce
   data, negative-controls-on, negative-controls-off, single-token
   answers.

## Next options

A. **Pivot paper to safe-consolidation-gates workshop story.** Track A's
   43 commits + 6 KL-drift rollbacks at AUC ≥0.85 is a real
   contribution. Workshop @ ICLR MemAgents 2026, ~4-6 weeks to ship.

B. **Try direct memory→logit shortcut.** Bypass all 28 layers; have a
   tiny MLP read memory bank values and add directly to lm_head logits.
   Forces the value into the output but changes the paper claim from
   "memory shapes prior-to-computation" to "memory is a retrieval
   head". ~4 hours to implement, ~1 hour training.

C. **Unfreeze Qwen.** Train Qwen UNFROZEN with LoRA + memory bank on
   memorize-recall. If memory routing works with full fine-tuning,
   isolate which layers need unfreezing to make it work on frozen.
   ~8 hours, expensive.

D. **Scale up to Qwen 7B or 14B.** Larger base may have more capacity
   to absorb cross-attn injection without dilution. ~6-12 hours train.

Pending Saam's decision.
