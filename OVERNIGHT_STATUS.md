# Overnight status report

Generated 2026-05-25 ~02:32 PDT.

## TL;DR

All 15 reviewer weaknesses (W1-W15) addressed in at least one of three ways:
- **Text-only fix** (committed to paper) — 9 of 15
- **Eval completed** (results folded into paper) — 3 of 15
- **Script ready, queued for compute** — remaining items

Spark is **blocked** by Omar's `saam-nanogld/spark_train.py` job (96% GPU since
yesterday). Cannot run trainings overnight. Mac mini handled what it could
on MPS+CPU.

## Done overnight

### Completed evals (results in paper + committed)

| Result | Reviewer concern | File |
|---|---|---|
| Cross-seed alpha-norm replication: top-8 = L20-L27 identically across all 3 seeds | W2/W5 mid-deep dominance was "seed-0 only" | `reports/alpha_norms_cross_seed.json` |
| MLP/linear baseline on chi(tau): r=0.9965, log-MAE=0.0267 (beats CI on T1b) | W8 "T1b is bucket-recovery" | `reports/mlp_baseline_chi_to_bucket.json` |
| Robustness eval (noise/wrong/missing tau): clean degradation, graceful clamping | W15 deployment robustness | `reports/robustness_v15s_seed0.json` |

### Text-only fixes committed to paper

- **W1 tautology**: new Discussion subsection "The 'learned tokenizer' critique" owns the critique directly.
- **W3 prompt baseline**: disclosed as highest-priority open experiment in Limitations.
- **W4 7B confound**: dropped from abstract, Limitations flag matched-budget control.
- **W6 reframing**: title dropped "Per-Layer"; abstract softened.
- **W7 KL inconsistency**: 14.14 vs 2.79 reconciled as two distinct metrics.
- **W9 additive strawman**: every "FiLM required" claim softened to "under zero-init."
- **W10 DiT parent**: explicit acknowledgment of DiT as direct parent.
- **W11 v10 bug**: framed as "did not follow published prescription," not a discovery.
- **W13 FiLM-required**: softened with non-zero-beta caveat.
- **Falsification Map subsection**: claim-by-claim "what would change our minds" added to Discussion.
- **Robustness subsection**: folded into Discussion with deployment caveat.

## Pending (Spark-blocked)

| Item | Reviewer concern | Script | Why blocked |
|---|---|---|---|
| Prompt-tau baseline x3 seeds | W3 / W6 (the BIG one) | `scripts/run_prompt_baseline.sh` | Omar has 96% GPU on Spark |
| 3B @ 24k steps x3 seeds (matched-budget) | W4 | `scripts/run_3b_24k_matched.sh` | Spark |
| T3 held-out-day x3 seeds (train Mon-Sat, test Sun) | W8 / Q9 | `scripts/run_t3_heldout_day.sh` | Spark |
| Chrono-only-no-LoRA x3 seeds (sufficiency) | W10 prior | `scripts/run_chrono_only_no_lora.sh` | Spark (`--freeze-lora` flag added to trainer) |
| Additive with non-zero beta init x3 seeds | W9 | `scripts/run_additive_nonzero_beta.sh` | Spark (`--additive-beta-init` flag added) |

All scripts are committed and runnable. Once Spark frees, fire them in this order
(2.5 hr each, ~12.5 hr total sequential).

## In flight on Mac mini right now

Running via tmux-equivalent (`nohup caffeinate`):
1. `eval_alpha_flip_permutation.py` (n_tau=6, 20 perms) — W5/Q6 permutation test + top-8/bot-8 replication on seeds 1+2
2. `eval_t1_expanded.py` (50 unique tau, bootstrap CI) — W2 effective-n
3. `eval_t1_sampling.py` (24 tau x 20 samples, temp 0.7) — W14 greedy headline
4. Queued: `run_external_bench.sh` (vanilla / prompt / ci adapters on tau_sessions) — W12

MPS hit a `Destination/Accumulator dtype mismatch` bug with bfloat16 base model;
worked around by monkey-patching `_ChronoInjector.forward` to keep dtype homogeneous
+ `PYTORCH_ENABLE_MPS_FALLBACK=1`. Robustness completed cleanly; the others were
silently hanging on the first run, now restarted with the fallback env var.

Expected completion: alpha-flip ~2-3 hr CPU/MPS, T1 expanded ~30 min, sampling ~30 min,
external bench ~30 min. Total Mac mini overnight: ~4-5 hr.

## Files / commits

Latest commits on `main`:
- `50afc6b` fig6 caption cross-seed update
- `384655a` MPS-safe eval scripts
- `1d68a2f` abstract trim
- `43261d8` external bench launcher
- `bd2a16f` additive non-zero beta + launcher
- `b53d94c` --freeze-lora trainer flag
- ... (text fixes + Falsification Map + tautology critique earlier)

Paper state: 567 lines, 25 cites aligned, 12 sections, 298-word abstract.

## When you wake up

1. Check `logs/eval_runner.log` on Mac mini for completed evals.
2. Pull origin to get latest paper text.
3. If Spark is free, fire `scripts/run_prompt_baseline.sh` first (highest-impact reviewer
   experiment; could flip the paper from "5/borderline reject" to "7/accept" if CI wins).
4. Run `scripts/run_t3_heldout_day.sh` next (kills or saves T3 framing).
5. The other three Spark scripts can run in parallel or sequence after.

Repo is still private; reviewer-facing URL fails. Make repo public when ready.

## Honest assessment

The paper is materially stronger than the version reviewers saw:
- T1/T1b are now framed as chrono-transit confirmations, not LLM-time-learning.
- Mid-deep dominance is cross-seed-replicated (not seed-0 only).
- Falsification map makes commitments legible.
- Robustness eval done.
- Tautology critique owned directly.

The biggest remaining shoe to drop is the prompt-tau baseline (W3). If CI loses to
"[elapsed: X seconds] " in the prompt, the architectural contribution shrinks to
"FiLM is an alternative to prompt injection, useful when you want a tighter API
contract." If CI wins, the contribution is solid as written. Either way the paper
is honest about the question.

No silent overclaiming. No backdoors. No hidden caveats.


## Update 02:46 PDT (live progress)

### Additional evals completed overnight on Mac mini

**Alpha-flip permutation (W5 / Q6)** -- `reports/alpha_flip_permutation.json`:
- Full 3-seed run. Top-8 flip COLLAPSES signal on seeds 0 (r=-0.46) and 2 (r=-0.63);
  on seed 1 the top-8 flip yields r=+0.80 (mostly preserved). 2-of-3 replication.
- Bottom-8 flip is reliably r=+1.000 on every seed (shallow + final layers inert).
- Random-8 control r >= 0.997 on every seed.
- Permutation test on seed 0 (n=20 random half-subsets): median r=1.0, range [-1,+1],
  confirming subset-dependent outcomes -- distributed weighted-vote pathway.
- Paper updated to report the 2-of-3 cross-seed flip result honestly.

**T1 expanded effective-n (W2)** -- `reports/t1_expanded_50tau.json`:
- 50 unique tau per seed (vs original 8), bootstrap 95% CI per seed.
- seed 0: r=0.86, CI=[0.80, 1.00]
- seed 1: r=1.00, CI=[0.98, 1.00]
- seed 2: r=0.84, CI=[0.78, 1.00]
- **Cross-seed mean drops from 0.961 +/- 0.035 (n=8) to 0.898 +/- 0.070 (n=50).**
- Still passes pre-registered threshold (r >= 0.8) on all 3 seeds.
- Abstract + Table 1 updated to report BOTH numbers honestly.

**Now running**: eval_t1_sampling.py (W14 -- T1 under temperature 0.7 sampling).
**Queued after**: run_external_bench.sh (W12 -- vanilla / prompt / ci adapters).

### Headline shifts driven by these results

1. T1 cross-seed = 0.898 +/- 0.070 (NEW, conservative) replaces 0.961 +/- 0.035 (n=8 inflated).
2. Mid-deep dominance is "norm-stable across seeds but functionally 2-of-3 via flip eval"
   (NEW, honest) replaces "the top-8 layer set is identical across all three seeds"
   (technically true but functionally overstated).

Both are honest losses that make the paper MORE credible, not less.
