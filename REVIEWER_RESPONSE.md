# Preemptive Reviewer Response

**Paper:** Chronometric Injection: Time-Conditional Behavior in a Frozen LLM via Per-Layer FiLM of Real Elapsed Seconds
**Repository:** https://github.com/sam-siavoshian/Time-Model
**Release pinned:** [`v15.0`](https://github.com/sam-siavoshian/Time-Model/releases/tag/v15.0)
**Date:** 2026-05-24

This document anticipates 15 hostile-reviewer attacks against the v15 paper and addresses each one with concrete evidence. The structure is: **attack stated in the reviewer's voice**, then **response** in 1-3 short paragraphs. Each response cites the exact section of [PAPER.md](PAPER.md), the JSON report in `reports/`, the script in `scripts/` or `model/`, and the numeric result.

Where an attack genuinely lands and we have no answer, the response says so plainly and points to the corresponding Limitations entry.

---

## 1. "All your metrics zero with α frozen, but the LoRA does the work."

**Reviewer voice.** Your headline T1 = 0.961 is meaningless without showing what a plain LoRA on the same data, same architecture, same budget achieves. The chrono channel may be decoration. The LoRA adapter, given 18 K conversations of clock readouts, has every reason to memorize the formatter vocabulary and produce τ-shaped outputs from prompt tokens alone.

**Response.** This is the single most important reviewer attack and the result that anchors the paper. We trained the v15 spec with `--freeze-alpha` so every per-layer chrono gate α is locked at 0 throughout training. The chrono encoder, the per-layer γ/β projectors, and all LoRA parameters still exist and receive gradients, but the chrono signal cannot reach the residual stream. Same 18 K conversations, same 18 K steps, same 15-scale encoder, same 50/50 phase balance.

Across 3 seeds (0, 1, 2), every single pre-registered test collapses to zero: T1 = 0.000 ± 0.000, T1b r = 0.000 ± 0.000, T1b log-MAE = 2.62 ± 1.02 (73× worse than v15), T2 = 0.000 ± 0.000, T3 = 0.000 ± 0.000, T4 = 0.000 ± 0.000. See [PAPER.md §24.7.8](PAPER.md) and the JSON reports `reports/qwen_time_lora_only_20260523_182213_recall.json` plus the two seed-1 and seed-2 reruns. The LoRA adapter, given identical training data, cannot leak the formatter vocabulary into a useful τ-conditional output. The chrono channel is load-bearing.

---

## 2. "T1 paraphrase generalization is just memorization of the formatter."

**Reviewer voice.** You report Pearson r = +0.996 on 10 paraphrased clock prompts. That number is suspiciously identical to 6 decimal places across phrasings as different as "How many seconds or minutes have gone by?" and "Time elapsed?". The model is almost certainly producing bit-identical outputs regardless of prompt, then you are computing r over the same fitted function.

**Response.** Half the attack lands and we have already reframed accordingly. Per-prompt Pearson r is in [0.99577, 0.99620] for all 11 phrasings (anchor + 10 paraphrases), mean +0.9960 ± 0.0001, on the v15 seed 0 checkpoint. We then added response-text logging and computed verbatim-identity rates between paraphrased responses and the anchor response at each τ. Mean fraction-identical across τ = 0.84. At mid-to-high τ (106 s, 10 659 s, 190 099 s, 318 283 s) the identity rate is 1.00.

The honest read is **not** "model generalizes to paraphrases" but **"model has a prompt-invariant, τ-conditioned formatter."** This is actually stronger evidence FOR chrono use, not weaker: the response is dictated by τ, not by prompt tokens. A LoRA-template-matching attack would predict that different paraphrases route through different LoRA-activated tokens and produce different outputs. They do not. See [PAPER.md §24.7.12](PAPER.md) and `reports/extra_controls_v2_v15s_seed0.json`. The original §24.7.9 framing has been corrected to make the response-identity finding explicit; the paraphrase result is reported as "prompt-invariant τ-driven formatter" rather than "paraphrase-generalizing."

---

## 3. "T2 and T3 with greedy decode have effective n=1."

**Reviewer voice.** You report T2 silent-gap Δ = 1.00 over n=30 trials and T3 weekend signal across n=20 trials. Under greedy decoding the model produces a deterministic output for each (prompt, τ) tuple. Your effective sample size is 1 per condition, not 30. The "perfect separation" reads as one prompt template plus one deterministic response.

**Response.** Attack accepted at face value, then resolved. We added explicit effective-n disclosure for every test in [PAPER.md §24.7.6](PAPER.md) (T1 reported n=64 → effective n=8 unique τ; T2 effective n=30 pairs over 1 fixed template; T3 effective n=2 unique prompts; T4 effective n=9 pairs over 3 prompts; α-flip effective n=3 parsed τ). The reported numbers are unchanged; the framing is precise about what they measure.

For T2 and T3 specifically, we then reran with temperature 0.7 over 30 independent torch seeds per condition (`model/qwen_time_t2t3_sampling.py`, `reports/t2t3_sampling_v15s_seed0.json`, [PAPER.md §24.7.13](PAPER.md)). T2: at τ=10s, ack_rate = 0.000 (0/30); at τ=86400s, ack_rate = 1.000 (30/30); Δ = +1.00, with 1 unique short-τ response and 3 unique long-τ responses ("21 hours", "22 hours", "23 hours") confirming real sampling variance. T3: weekend signal = +0.833 ± 0.379, weekday signal = +0.433 ± 0.626 across 30 real seeds with 3-4 unique responses per condition. Both still pass thresholds (T2 ≥ 0.5, T3 ≥ 0.3) with genuine, non-trivial n=30. Attack does not land.

---

## 4. "Single coherent dial framing is overstated."

**Reviewer voice.** You sell the α-sign-flip r = −0.9998 as evidence that "the chrono signal is one causal scalar dial." But you only ran the all-layer flip. If chrono were genuinely one scalar dial, ANY half-layer-flip subset should collapse the signal to zero (half the votes flip → cancellation). You never tested that.

**Response.** Attack lands and we ran the experiment. `model/qwen_time_extra_controls.py` runs 5 conditions on v15 seed 0: A normal, B all-flipped (NaN: model output broken), C half flipped with rng seed 42 (r = +0.7842, sign preserved), D half flipped with rng seed 7 (r = −0.9336, sign inverted), E one-third flipped (r = +0.8936, sign preserved). Different random 17-layer subsets give r = +0.78 vs −0.93. The single-coherent-scalar interpretation is too strong.

We then dumped per-layer α norms and ran a targeted flip ([PAPER.md §24.7.11](PAPER.md), `reports/alpha_norms_v15s_seed0.json`). Top-8 dominant layers are L26, L23, L25, L20, L24, L22, L21, L27 (mid-deep block L19-L28); bottom-5 least dominant are L7, L11, L10, L8, L9 (shallow). Flipping just the top-8 dominant layers drops T1 r from +1.000 to −0.18 (signal vanishes, does not invert). Flipping the bottom-8 leaves r = +0.9998 untouched. Random-8 gives r = +0.95.

The corrected framing in [PAPER.md §24.7.9, §25 closing paragraph](PAPER.md) is: **"The chrono signal acts as a weighted sum of per-layer monotone-in-τ contributions with non-uniform layer dominance. Different layers contribute different magnitudes; specific dominant mid-deep layers determine the overall sign. Flipping ALL alphas at once cleanly inverts because every per-layer vote flips together (r = −0.9998 on n=3 unique τ), but flipping random subsets produces variable outcomes depending on which subset contains the dominant layers."** This is still a strong causal claim. It is not "one knob." It is "many knobs that vote, with a dominant subset."

---

## 5. "Probe -143 R² is broken."

**Reviewer voice.** Your probe collapses to R² = −143 in the α=0 condition. R² that negative means the regressor is producing wild numerical garbage, not "the signal is destroyed." You are pattern-matching a numerical pathology and calling it evidence.

**Response.** Attack lands and is now explicitly conceded. The −143 is a ridge-solver pathology on standardized features with degenerate variance. We added prediction clamping to `[y_train.min(), y_train.max()]` in `model/qwen_time_probe.py` and reran on v15 seed 0 (`reports/probe_v5_clamped_v15s_seed0.json`, [PAPER.md §24.7.14](PAPER.md)).

After clamping, condition A (trained) gives R² = −2.42 uniformly across layers (within-distribution interpolation collapses on the OOD split; the original v11 +0.43 was a within-distribution luck draw); condition B (α=0) still shows R² = −143 across all layers (the clamp does not change the chrono-off hidden states being uniformly degenerate-conditioned for ridge); condition C (shuffled) gives R² = +0.027. The 140-point gap between A and B IS still meaningful (chrono-off hidden states are catastrophically worse-conditioned than trained ones), but **the absolute R² number should not be reported as "tau lives in the residual stream."** A within-distribution probe split and a Spearman rank-correlation metric would be more defensible. We list this as future work and explicitly retract the deep-layer linear-decodability claim.

What survives: tau is linearly decodable at L1-L3 within distribution (R² peak +0.43 at L1) on the v11 anchor; chrono-off hidden states are pathologically degenerate. What does not: any claim about OOD linear extrapolation of tau from the residual stream beyond layer 3.

---

## 6. "Per-layer is not novel vs just-L0."

**Reviewer voice.** Your contribution is "per-layer AdaLN-Zero FiLM injection." But you never showed per-layer beats single-layer. Inject only at layer 0, train the same way, see what happens. I bet it does the same job.

**Response.** Attack runs the experiment. `bash scripts/run_ablation_l0_only.sh` trains v15 spec with `--inject-layers 0`. Result on seed 0, in `reports/qwen_time_l0_only_20260524_055911_recall.json`, table in [PAPER.md §24.7.10](PAPER.md):

| Variant | T1 | T1b r | T1b log-MAE | T2 | T3 weekend | T4 |
|---|---|---|---|---|---|---|
| v15 (FiLM, every layer) | 0.950 | 0.994 | **0.032** | 1.00 | 1.00 | **0.272** |
| L0-only (FiLM, layer 0) | **1.000** | 0.989 | 0.137 | 1.00 | 1.00 | 0.018 |

L0-only matches v15 on 4 of 5 tests, even exceeds on T1. T1b log-MAE degrades ~4× (0.032 → 0.137), and T4 KL drops ~15× (0.272 → 0.018). The honest architectural contribution is narrowed in [PAPER.md §24.7.10](PAPER.md): **"AdaLN-Zero FiLM modulation of a continuous wall-clock encoding at ≥1 decoder layer, with DiT-style γ-bias=1 init."** Per-layer placement is a precision optimization that buys T1b precision and stronger output-side chrono routing, not a categorical requirement. This narrows the novelty claim relative to Timely Machine (2601.16486, token-level scaling) but does not eliminate it; the FiLM-and-DiT-init combination is still the distinguishing piece, not the per-layer placement.

---

## 7. "FiLM is not novel vs additive."

**Reviewer voice.** You claim FiLM `h + α(γh + β)` is load-bearing. Why? Plain additive `h + α · W_χ χ` should do the same job. Same trainable count, same chrono signal, simpler math. I bet additive matches.

**Response.** Attack runs the experiment. `bash scripts/run_ablation_additive.sh` trains v15 spec with `--injection-type additive`. Result on seed 0, in `reports/qwen_time_additive_20260524_070829_recall.json`, table in [PAPER.md §24.7.10](PAPER.md):

| Variant | T1 | T1b r | T1b log-MAE | T2 | T3 weekend | T4 |
|---|---|---|---|---|---|---|
| Additive every-layer (NO FiLM) | **0.000** | **0.000** | 1.86 | **0.00** | **0.00** | **0.000** |
| LoRA-only (no chrono) | 0.000 | 0.000 | 3.20 | 0.00 | 0.00 | 0.000 |

Additive collapses to the same all-zero pattern as LoRA-only. The mechanistic explanation in [PAPER.md §24.7.10](PAPER.md): under α=0 / β-bias=0 init, the gradient through α is `∂out/∂α = β = 0` at init, so α cannot move from zero. FiLM escapes this because `∂out/∂α = γh + β = h ≠ 0` at init (γ-bias=1). This is the same trainability mechanism that killed v10 (γ-bias=0, dead) and motivated the v10→v11 DiT-init fix in [PAPER.md §23.1](PAPER.md). **The FiLM design is mathematically necessary at this init pattern, not aesthetic.** Distinguishes from GazeQwen (2603.25841, additive residual of a different continuous signal) which we predict would fall in the same init trap absent a γ-bias-1 fix.

---

## 8. "T3 with 0.667 ± 0.577 std is meaningless."

**Reviewer voice.** You report T3 weekend signal = 2 of 3 seeds pass and frame it as "fragile under seed randomness." That's a binary outcome at n=3. You don't have variance; you have a coin flip with three trials. The paper's table line "0.667 ± 0.577" treats a Bernoulli as Gaussian.

**Response.** Attack is correct and we have already corrected the framing. [PAPER.md §24.7.5](PAPER.md) and the headline table in [README.md:42](README.md) report T3 as "**2 of 3 seeds pass** (signal ∈ {1.0, 0.0, 1.0}; seed 1 mode-collapses)" rather than as a continuous mean ± std. The "0.667 ± 0.577" Gaussian-style summary appears nowhere in the body claims; it is mentioned in [PAPER.md §24.7.6](PAPER.md) only to explicitly disclaim it.

The honest characterization: T3 is a binary per-seed outcome on 2 fixed eval prompts (Wednesday + Saturday). Two of three v15 seeds discriminate weekend from weekday with weekend_signal = 1.0; one seed mode-collapses to a single default response template. The mode collapse is a real training-fragility finding, not a Gaussian tail. We list T3 as "fragile" in the headline result table and as a Limitation in [PAPER.md §26.2](PAPER.md). Future work to make T3 reliable: more phase data, longer training, or curriculum.

---

## 9. "Title overpromises 'experience time'."

**Reviewer voice.** Your BibTeX title in CITATION.cff says "Teaching a Frozen LLM to Experience Time." This is a behavioral result. "Experience" is a phenomenological claim. You have no evidence the model experiences anything. Strip the language.

**Response.** Attack lands and was already addressed. The paper title was rewritten to: **"Chronometric Injection: Time-Conditional Behavior in a Frozen LLM via Per-Layer FiLM of Real Elapsed Seconds."** The "experience time" phrasing remains only in the [README.md:17 callout](README.md) and the legacy BibTeX in [README.md:262](README.md). The paper body, [PAPER.md §25 conclusion](PAPER.md), and [PAPER.md §26.3 one-paragraph headline](PAPER.md) all use "time-conditional behavior" with explicit disclaimers against subjective-experience interpretations (see [README.md:201](README.md): "Qualia or experiential time. This is a behavioral and mechanistic result, not a phenomenological one"). We will update the BibTeX title before final submission to match the paper title. Berg et al 2510.24797 is cited as the relevant adjacent work on whether time-conditional behavior implies anything about experience (see [PAPER.md §13.1, §25.1](PAPER.md)).

---

## 10. "OOD transfer claim was inflated."

**Reviewer voice.** Your abstract used to claim "OOD task transfer to deadline-induced response length" with chrono-only +9 tokens. The supporting evidence was n=5 with max_new=80 right-censoring three of those five responses. The +9 was carried by one outlier prompt and three censored zeros. That is not statistical evidence of anything.

**Response.** Attack lands completely. The claim is **retracted** in the current draft. We ran `model/qwen_time_pressure_v2.py` with n=30 neutral prompts, max_new=256 (uncensored), bootstrap 95 % CI on paired deltas. Result on v15 ([PAPER.md §24.7.3](PAPER.md), `reports/qwen_time_v15_*_pressure_v2.json`):

| Condition | mean delta (tokens) | 95 % CI | CI excludes 0? |
|---|---|---|---|
| P1 (text deadline + matching τ) | +76.5 | [+51, +104] | yes |
| **P2 (chrono only, neutral text)** | **+3.4** | **[−16, +22]** | **no** |
| P3 (α = 0 + text deadline) | +121.4 | [+89, +151] | yes |
| Chrono contribution (P1 − P3) | **−44.8** | [−80, −9] | yes on the **negative** side |

Worse than just-fails-to-replicate: chrono actively **attenuates** the text-deadline length shift by ~45 tokens. We have removed the OOD-transfer claim from the abstract, contributions list, and the §24.2 verdict. The new framing in [PAPER.md §24.7.3, §25, §26.2](PAPER.md): "The chrono signal trained on clock readout, silent-gap acknowledgment, and weekly phase produces measurable in-distribution behavioral effects but does **not** transfer constructively to deadline-induced response-length modulation; on the contrary, in v15 the chrono signal slightly attenuates the response a text deadline would otherwise produce." Workshop strength only on the OOD axis. We report this ourselves rather than letting a reviewer find it.

---

## 11. "Why not test on existing benchmarks like TimeBench?"

**Reviewer voice.** You evaluate on five tests you designed yourself. Convenient. There are existing public benchmarks for temporal reasoning in LLMs: TimeBench, BombRush (Wang et al 2506.05790), Timely-Eval (Ma et al 2601.16486). Run on those and let us compare apples to apples.

**Response.** Attack is partially fair, partially missing the point. No public benchmark we are aware of (BombRush, Timely-Eval, TimeBench, the 76-scenario suite from Cheng et al 2510.23853) injects **a real-valued elapsed-time tensor into the model's forward pass.** They all evaluate prompt-encoded time (the model receives "3 hours ago" as tokens) or test-time scaling (the model decides how many tokens to spend). Our architectural claim is specifically about a continuous τ input that the prompt does not contain. The 5 pre-registered tests in [PAPER.md §23.9](PAPER.md) are designed to evaluate that specific input modality.

That said, the attack lands on lack-of-comparison-to-prompt-only-baselines. We do not currently run BombRush or Timely-Eval against vanilla Qwen 2.5 3B vs CI-Qwen. This is honestly listed as future work in [README.md:53](README.md) ("No external-benchmark validation; future work") and in [PAPER.md §25.1](PAPER.md). We are releasing our 5-test suite as a public benchmark suite with the v15 release so others can evaluate against the same protocol. The eval harness is one command (`uv run python -m model.qwen_time_check --checkpoint <ckpt> ...`); reproducible from the released v15.0 checkpoints with no training required.

---

## 12. "n=3 seeds is too few."

**Reviewer voice.** Your headline numbers are mean ± std over n=3 training seeds. That's not a sample, that's a triplet. Standard practice is n=5 minimum, n=10 for ML papers with seed sensitivity. With n=3 a single outlier shifts the mean by 33% and the variance is uninterpretable.

**Response.** Attack is statistically correct. We disclose the budget honestly: each v15 seed takes ~45 min on a single GB10; 3 seeds × 45 min = 2.25 GPU-hours, all in-repo. Going to n=10 would multiply that by ~3.3 and increase total v15-era compute from ~10.9 to ~26 GPU-hours. This is in scope for a v2 of the paper and is listed in [PAPER.md §26.4](PAPER.md) future work. The reason it has not happened yet is the compute box (DGX Spark prototype) is shared with another researcher and we coordinate GPU access (see `~/.claude/CLAUDE.md` Spark section).

Two mitigations applied. (1) Where n=3 is too few to be meaningful (T3, a binary outcome), we report the raw count instead of a Gaussian summary ("2 of 3 seeds pass" not "0.67 ± 0.58"). (2) The variance bars where they ARE reported are tight on the metrics that matter: T1 = 0.961 ± 0.035, T1b r = 0.993 ± 0.003, T2 = 1.00 ± 0.00 (saturated), T4 multi-pos KL = 14.14 ± 1.15. The signal is large relative to inter-seed variance. The α-flip causal intervention (r = −0.9998) is single-seed on v15 seed 0; this is flagged as a limitation in [PAPER.md §26.4](PAPER.md). Cross-seed α-flip is the highest-priority follow-up.

---

## 13. "T3 fails periodicity beyond 14 days."

**Reviewer voice.** Your T3 weekday/weekend signal passes at week 1 (trained, τ = 5.5 d) and week 2 (OOD, τ = 12.5 d). Then it dies: week 3 (19.5 d) signal = 0, week 4 (26.5 d) signal = 0. You claim periodicity but you only demonstrate generalization for one period.

**Response.** Attack is empirically correct on the surface, but the root cause is in the training distribution, not the architecture. [PAPER.md §24.7.2](PAPER.md) (round-2 reframe added 2026-05-24): training τ for the phase task is drawn uniformly in `[0, 7 d)`, the model saw **exactly one period of the weekly sinusoid** during training. With one period of exposure, the model has no example data from which to learn that the weekly sin/cos function is genuinely periodic. It only saw the function evaluated once across its domain. T3 multi-week "failure past week 2" is therefore **not** a generalization failure of a learned weekly phase representation; it is the predictable consequence of training τ ∈ [0, 1·T_week] failing to teach periodicity.

The chrono encoder mathematically computes the correct sin/cos value at any τ (the 604 800 s timescale is built in); the model's learned readout of that periodicity has been trained on one period only. To actually test weekly-phase generalization, training τ should span ≥ 3 weeks. Until that experiment is run, the result in [PAPER.md §24.7.1 multi-week table](PAPER.md) is reported as: "the model successfully maps τ → weekday/weekend within the trained week and the immediately adjacent week; degrades thereafter, consistent with the model having no chance to learn weekly periodicity from one-period-of-exposure training data." Future work explicitly named in [PAPER.md §25.1](PAPER.md): retrain with phase τ ∈ [0, 3·T_week] and rerun the multi-week test.

---

## 14. "What about reproducibility?"

**Reviewer voice.** This kind of paper falls over in peer review when nobody can reproduce the numbers. Where is your reproducibility checklist? Where are the checkpoints, the SHA hashes, the deterministic data pipeline, the dependency lock?

**Response.** Everything you ask for is in the repo. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) (the full NeurIPS-style checklist with one pointer per item). Specifically: (a) Trained checkpoints for all 3 v15 seeds released as GitHub Release [`v15.0`](https://github.com/sam-siavoshian/Time-Model/releases/tag/v15.0), with SHA256 hashes in [README.md:136-140](README.md). (b) Training data SHA256 in [data/VERIFICATION.md:13-16](data/VERIFICATION.md) for all 3 seed-specific JSONL files. (c) Dependency lock in [uv.lock](uv.lock); Python 3.11+ pinned in [pyproject.toml:7](pyproject.toml). (d) One-command reproduction for every headline number in [README.md:77-97](README.md). (e) End-to-end CPU smoke test (`bash scripts/e2e_smoke.sh`, ~10 min on M-series) verifies the pipeline executes without GPU.

For the headline cross-seed table specifically: `bash scripts/run_v15_seeds.sh && uv run python scripts/aggregate_seeds.py`. ~2.25 GPU-hours on a single GB10 or H100 or A100 80 GB. Tolerance bound: numbers match within ±0.02 on Pearson r, ±5 % on KL. Wider drift means a real difference, not noise (this came up once when a hardware swap from one Spark to another shifted T4 by 0.4 %; within tolerance). If a paper number has no command in [README.md](README.md), that is filed as a reproducibility bug, not a missing feature.

---

## 15. "You only tested Qwen 2.5 3B. Why should I believe this generalizes?"

**Reviewer voice.** Your headline numbers are on Qwen 2.5 3B. You attempted 7B and got T1 = 0.747 (below threshold). You tried 14B and OOMed. Where is the evidence the architectural claim transfers to (a) other base sizes, (b) other base families (Llama, Mistral, GPT), (c) other modalities? Without that, this is a single-checkpoint result dressed up as an architectural contribution.

**Response.** Attack is partially fair, partially answered. Partial answer: the 7B run ([PAPER.md §24.6.4](PAPER.md)) reproduces 4 of 5 properties (T1b OOD r = 0.76, T2 = 1.00, T4 KL = 0.129, actually stronger than 3B); T1 in-distribution at 0.747 is below the 0.8 threshold most likely because 12 K steps is undertrained for 7B (3B's analogous result was 0.94 at 12 K steps; the v15 jump to 18 K steps lifted 3B to 0.961 ± 0.035). The architectural pattern holds at 7B; the training budget needs scaling. 14B OOMs on 128 GB GB10 unified memory at step 1146; requires FP8, model parallelism, or smaller LoRA rank, all out of scope for this paper.

Other base families: not tested. This is honestly disclosed as a limitation in [README.md:203](README.md) and [PAPER.md §25.1](PAPER.md) ("Generalization to other base models. Only tested on Qwen 2.5 3B-Instruct. Other bases are future work"). Other modalities: not tested ([PAPER.md §25.1](PAPER.md): "audio and video models could plausibly use the same chronometric injection over their existing positional encodings. Untested.")

The claim in the paper is therefore scoped: per-layer AdaLN-Zero FiLM injection of a 27-dim sinusoidal+log encoding of real elapsed seconds **on Qwen 2.5 3B**, with the four-property survival pattern reproducing on 7B at lower precision. We are not claiming universal architectural validity. The architecture is small, simple, reproducible in ~45 min on a single GPU per seed; if a reviewer believes the claim should generalize, they can run it on their preferred base in one afternoon and tell us. This is in scope for a v2 paper, not for the current submission.

---

## Items where the attack would land if raised

These are not in the 15 above because the parent prompt did not ask, but for completeness we note attacks the current paper genuinely does not answer:

- **"Why don't you have a non-chrono temporal-input baseline?"** We compare to LoRA-only (zero chrono) and to architectural ablations (additive, L0-only). We do NOT compare to a "prompt-injected τ-as-tokens" baseline where the same number "τ = 12 478 seconds" is fed via the prompt. This is a real gap. Listed as future work; no current experiment.
- **"Mechanistic-interpretability claim past layer 3 is unsupported."** The MLP probe overfits ([PAPER.md §24.3 update](PAPER.md)); clamped linear probe also fails on OOD splits ([PAPER.md §24.7.14](PAPER.md)). We have no positive evidence for how deep layers represent τ; we only know that the α=0 intervention makes deep-layer hidden states catastrophically degenerate. This is a real limitation, conceded in [PAPER.md §24.7.14](PAPER.md) and [PAPER.md §25.1](PAPER.md) item 2.
- **"Persistence under no-input is behavioral, not stateful."** T2 silent-gap is a behavioral test of gap awareness, not a test that the model's internal state evolves during a real wall-clock gap. A forward pass is instantaneous. The deeper version of "the model experiences elapsed time during silence" requires continuous-time recurrence between forwards (Neural ODE), which we have not implemented. Conceded explicitly in [PAPER.md §25.1](PAPER.md) item 4.
