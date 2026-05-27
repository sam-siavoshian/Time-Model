# PREREGISTRATION_v2 — Chronometric Injection (CI) revision

**Project:** Chronometric Injection (CI), submission v2 revision.
**Author:** Saam Siavoshian.
**Date locked:** 2026-05-26.
**Repository:** https://github.com/sam-siavoshian/Time-Model
**Supersedes:** PREREGISTRATION.md (2026-05-12, IPCN-era, now deprecated).

This document locks the hypothesis, sample, inclusion rule, statistic,
threshold, decision rule, and analysis script for every new experiment in
the v2 revision BEFORE any new training, sweep, or analysis is run.
Once this commit lands on `main`, no scenarios, metrics, lexicons,
checkpoints, decoding parameters, or analysis code paths used by the
pre-registered tests will be modified prior to saving the sweep outputs.
Amendments require a new commit appending to this file with an
`AMENDMENT YYYY-MM-DD` header and a justification.

Anchor commit: this file plus all referenced code is pushed at the same
commit; the paper cites that commit hash as the pre-registration anchor.

---

## 1. Experiment A: TPDR replication on 200 scenarios

### 1.1 Hypothesis (directional)

On the `deliberative_score` metric (from `eval/tpdr/metrics.py`,
`DELIBERATIVE_LEXICON_HI`), the paired difference of per-scenario Pearson
correlation `r(metric_value, log(tau))`, computed as `CI - Prompt`, has a
negative population mean. The CI adapter exhibits a more strongly
negative deliberative-score elasticity than the Prompt adapter, i.e. CI
sheds deliberative-vocabulary tokens faster as elapsed time increases.

- H1 (directional): `mu_diff < 0`
- H0: `mu_diff >= 0`
- Test: one-tailed paired Student's `t`-test on per-scenario diffs
  (`scipy.stats.ttest_rel`, then halve the two-sided p for the directional
  alternative iff `t < 0`; report `t`, df, exact two-sided p, one-tailed
  p, and 95% CI on the mean diff).

### 1.2 Sample

- All 200 scenarios as committed at HEAD in `eval/tpdr/scenarios.py`.
- `assert len(SCENARIOS) == 200` is in that file at the anchor commit.
- No scenarios will be added, removed, or modified before the sweep
  outputs `reports/tpdr_crossseed/tpdr_v2_seed{s}_pair{p}.json` are
  saved.
- The 7 metrics and their 4 lexicons in `eval/tpdr/metrics.py` are
  frozen at the anchor commit. No metric will be added, removed, or
  modified before the sweep outputs are saved.

### 1.3 Inclusion rule (per-scenario)

A scenario contributes a pair `(r_ci, r_prompt)` to the paired-t sample
on the `deliberative_score` metric iff BOTH of the following hold:

1. The `deliberative_score` series across the 10 `tau` values has
   non-zero variance for the CI adapter on that scenario, AND
2. The same series has non-zero variance for the Prompt adapter on that
   scenario.

Constant-across-tau series yield `nan` Pearson `r` and are excluded.
The `pearson_safe` function in `eval/tpdr/metrics.py` (anchor commit)
returns `nan` when either variance is zero; this is the exclusion rule.

### 1.4 Sample-size prediction

The original 50-scenario sweep observed `n_paired = 11` on
`deliberative_score`, an inclusion rate of 22.0%. Linearly extrapolating
to 200 scenarios predicts `E[n_paired] approx 44`.

Decision rule on observed `n_paired`:

- `n_paired >= 25`: test is sufficiently powered; report the result as
  pre-registered confirmatory evidence (positive or null).
- `n_paired < 25`: test is underpowered; report as **inconclusive**, not
  as confirmatory evidence. Headline claim must not rest on this number.

### 1.5 Primary endpoint

- Metric: `deliberative_score`.
- Statistic: one-tailed paired Student's `t` on per-scenario `(r_ci -
  r_prompt)` differences.
- Significance level: `alpha = 0.005` (one-tailed).

The tighter `alpha = 0.005` corrects for the implicit one-of-seven
metric-selection that produced the original headline. With seven
candidate metrics and an uncorrected `alpha = 0.05`, the family-wise
error rate is approximately `1 - 0.95^7 = 0.30`. Setting `alpha =
0.005` gives a Bonferroni-style one-of-seven adjustment that holds the
headline metric to a substantially stricter standard than the original
exploratory sweep.

The headline pair is `(ci_seed=0, prompt_seed=0)`. Cross-seed
replications are pairs `(1,1)` and `(2,2)`.

### 1.6 Secondary endpoints (exploratory, Holm family)

The remaining 6 metrics from `eval/tpdr/metrics.py` are pre-registered as
**exploratory secondary endpoints**:

`length_chars`, `length_words`, `urgency_score`, `hedge_score`,
`conditional_clauses`, `imperative_count`.

Family-wise Holm-Bonferroni correction at `alpha = 0.05` across `m = 6`
hypotheses. Two-tailed paired Student's `t` on per-scenario diffs.
These are NOT part of the headline claim. Sign of effect is not
pre-registered for the secondary family (i.e. these are two-tailed).

### 1.7 Seeds, decoding, hardware

- CI seeds: 0, 1, 2. Checkpoints:
  `release_ckpts/qwen_time_v15s_20260523_141410_seed0.pt`,
  `release_ckpts/qwen_time_v15s_20260523_141410_seed1.pt`,
  `release_ckpts/qwen_time_v15s_20260523_141410_seed2.pt`.
- Prompt-baseline seeds: 0, 1, 2. Checkpoints:
  `checkpoints/prompt_baseline_s0.pt`,
  `checkpoints/prompt_baseline_s1.pt`,
  `checkpoints/prompt_baseline_s2.pt`.
- Decoding: greedy, `max_new = 150` tokens, system prompt + chat
  template per `eval/tpdr/run_tpdr.py` at the anchor commit (no system
  prompt overrides, no sampling parameters changed).
- Hardware: NVIDIA GB10 (DGX Spark, aarch64). CUDA path in run_tpdr.py
  (`--device cuda`).

### 1.8 Vanilla baseline diagnostic (pre-sweep)

The reviewer flagged that the vanilla baseline emitted an identical
713-character template across all scenarios at all `tau` in the 50-scen
sweep. Before the main 200-scen sweep runs to completion, the vanilla
adapter is re-run on 10 random scenarios under two configurations:

- (a) Current configuration as in `eval/tpdr/run_tpdr.py` HEAD (greedy,
      `max_new = 150`, `<|im_start|>user ... assistant` chat template,
      no system prompt). Result saved to
      `reports/tpdr_vanilla_diagnostic_curconf.json`.
- (b) Stripped configuration: no chat template (raw `scenario` text),
      default greedy decoding, `max_new = 150`. Result saved to
      `reports/tpdr_vanilla_diagnostic_strip.json`.

Decision rule:

- If (a) and (b) both still produce a near-identical response across
  scenarios and `tau`, the template behavior is a property of
  Qwen 2.5 3B-Instruct on these prompts (not a configuration bug), and
  the existing vanilla numbers in §9 of the paper remain valid as a
  reference point.
- If (b) produces materially different (varying) responses while (a)
  produces an identical template, the original vanilla numbers are
  invalid and `run_tpdr.py` must be patched to a working configuration
  before the main sweep is launched.

### 1.9 Analysis script

The headline number is computed by
`eval/tpdr/analyze_v2.py` (added at the anchor commit, see below).
This script reads the saved sweep JSONs and outputs
`reports/tpdr_v2_headline.json` with:

- per-(adapter, metric) per-scenario `r` values for each seed pair
- the paired-t result on `deliberative_score` for the headline pair
  `(ci0, pr0)` (t, df, exact two-sided p, one-tailed p, 95% CI on
  mean-diff)
- `n_paired` for the headline pair
- the same numbers for the cross-seed replications `(ci1, pr1)` and
  `(ci2, pr2)`
- Holm-Bonferroni corrected p-values on the 6 secondary metrics for the
  headline pair

### 1.10 Decision rules (paper update)

After the analysis script is run:

- **CONFIRM**: `n_paired >= 25` AND one-tailed `p <= 0.005` on the
  headline pair. Headline the result as pre-registered confirmation of
  differential behavioral modulation. Cross-seed replications are
  reported alongside (positive, null, or mixed; reported honestly).
- **NULL**: `n_paired >= 25` AND one-tailed `p > 0.005`. Report the null
  result in the abstract and discussion. Remove any
  architectural-distinction claim from the abstract and introduction
  that rests on TPDR. Surviving contribution is mechanistic
  interpretability.
- **INCONCLUSIVE**: `n_paired < 25`. Report as inconclusive in the
  paper; do not headline. State that broader lexicons or a larger
  scenario set are required.

---

## 2. Experiment B: per-layer probe on clock-heldout checkpoints

### 2.1 Hypothesis (directional, two-way)

Before running: the per-layer probe `R^2(L=1)` on the clock-heldout
checkpoints either remains near 1.0 (indicating mechanical FiLM
readout regardless of downstream training task) or degrades meaningfully
(indicating L1 emerges from model-side processing trained on
CLOCK).

Pre-registered thresholds on `R^2(L=1)`, averaged across seeds 0, 1, 2
of `clock_heldout`:

- `>= 0.99`: mechanical readout. Transmission claim in §5.2 is demoted
  from "established mechanically" to "FiLM injection puts a
  linearly-decodable `tau` axis into the residual stream by
  construction".
- `< 0.95`: model-side dependence. Transmission claim is strengthened;
  L1 R^2 reflects acquired model-side use of the channel.
- `[0.95, 0.99)`: intermediate. Reported descriptively; no claim change.

### 2.2 Sample

Three clock-heldout checkpoints `clock_heldout_s{0,1,2}` already trained
in §8.1. No retraining.

### 2.3 Procedure

For each clock-heldout seed, run the probe at every layer
`L = 0, 1, ..., 36` using the same probe script that produced
`reports/probe_per_layer_v15s_s0.json` (full-supervision baseline).

Outputs: `reports/probe_per_layer_clock_heldout_s{0,1,2}.json`.

The same probe class, same fitting procedure, same residual extraction
points, same eval set as the v15s baseline. Code path is whatever
script produced `probe_per_layer_v15s_s0.json` (do not change it).

### 2.4 Paper update

Add `figures/fig_probe_clock_heldout.png` overlaying the four curves
(v15s_s0 + clock_heldout_s{0,1,2}) for L0 to L36. Add the
interpretation paragraph in §5.2 conditional on the threshold band.

---

## 3. Experiment C: held-out CLOCK + SILENT-GAP + ablation rows

### 3.1 Hypothesis (Experiment C.1 — both-heldout training)

Training data mix `clock_fraction = 0.0`, `silent_fraction = 0.0`,
`phase_fraction = 1.0` (PHASE-only) at 18k steps, n=3 seeds.

Pre-registered prediction:

- T1 (point-clock recall) and T1b (interpolation): both should produce
  `nan` recall (no clock supervision to learn the readout from).
- T2 (silent-gap acknowledgment): should collapse to `Delta approx 0`
  (no silent-gap supervision).
- T3 (phase recall): should pass at typical full-supervision levels
  (PHASE is the only task seen).
- T4 multi-position KL: likely degraded relative to full supervision
  (no clock/silent-gap shape to push the channel through), reported
  descriptively without a fixed threshold.

Outputs: `reports/both_heldout_s{0,1,2}_recall.json`.

### 3.2 Procedure

- Add `scripts/run_both_heldout.sh` and the data generator that
  produces a PHASE-only training jsonl.
- Train at 18k steps, n=3 seeds, identical hyperparameters to the
  full-supervision CI training in `scripts/run_v15s.sh` (or whichever
  is canonical for the released CI checkpoints).
- Evaluate with `qwen_time_check.py` using the standard T1/T1b/T2/T3/T4
  protocol.

### 3.3 IA3-only n=3 (Row 8 + Row 8-companion)

Current state: `ia3_only_s0_recall.json` exists locally; `_s1` is
running on Spark; `_s2` is queued. At pre-registration time the
prediction is that IA3-only (no chrono channel) produces zero T1/T1b/T2
recall (matches LoRA-only collapse already reported). When all three
seeds land, add the row to Table 3 at `n=3`.

### 3.4 Table 3 ablation rows (paper update)

Three rows are added to the paper's Table 3 ablations with all
T1/T1b/T2/T3/T4 columns populated and cell-level seed counts:

- **Row 1b**: chrono-only-without-LoRA, n=3 (s0, s1, s2). Source:
  `reports/chrono_only_s{0,1,2}_recall.json`.
- **Row 2b**: additive-with-`beta`-bias-init = 0.01, n=3 (s0, s1, s2).
  Source: `reports/additive_nonzero_beta_s{0,1,2}_recall.json`.
- **Row 8**: IA3-substituted PEFT with chrono active, n=3 (s0, s1, s2).
  Source: `reports/ia3_with_chrono_s{0,1,2}_recall.json`. (Already
  reported in prose; this entry locks the JSON-backed row.)

If `ia3_only_s2` does not complete by the paper submission deadline,
the row label degrades to its real `n`, with the count stated in the
row label (e.g., "IA3-only, n=2").

---

## 4. Operating rules during the revision

1. No new metric, scenario, lexicon, or scoring rule is added to the
   pre-registered tests between this commit and the saved outputs.
2. No experiment result is fabricated. If a run fails or cannot
   complete on available hardware in the available time, the paper says
   "not run" and explains why.
3. Honest reporting beats positive reporting. A clean null is more
   valuable to credibility than a marginal positive.
4. Every new claim in the paper cites the JSON report by path.
5. The paper text update cites this PREREGISTRATION_v2.md by commit
   hash.

---

## 5. Files anchored at this commit

- `PREREGISTRATION_v2.md` (this file)
- `eval/tpdr/scenarios.py` (200 scenarios, frozen)
- `eval/tpdr/metrics.py` (7 metrics, 4 lexicons, frozen)
- `eval/tpdr/run_tpdr.py` (greedy `max_new=150` decode, frozen)
- `eval/tpdr/analyze_v2.py` (analysis script, added at this commit)
- `scripts/run_both_heldout.sh` (training script, added at this commit;
  uses existing `model.qwen_time_data` with `--mix "0,0,1"` for the
  PHASE-only generator, no new data module required)
- `scripts/run_tpdr_vanilla_diagnostic.sh` (vanilla diagnostic, added
  at this commit)

End of preregistration v2.
