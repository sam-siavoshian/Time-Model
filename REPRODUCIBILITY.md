# Reproducibility Checklist

**Paper:** Chronometric Injection: Time-Conditional Behavior in a Frozen LLM via Per-Layer FiLM of Real Elapsed Seconds
**Author:** Saam Siavoshian
**Repository:** https://github.com/sam-siavoshian/Time-Model
**Release pinned for this checklist:** [`v15.0`](https://github.com/sam-siavoshian/Time-Model/releases/tag/v15.0)
**Date:** 2026-05-24

This document follows the NeurIPS 2025 Reproducibility Checklist (the most recent published version at submission time). One row per item. Each row gives a YES / NO / N/A verdict and a concrete pointer (file:line, release URL, JSON path, or section reference).

If anything in this table is wrong, that is a reproducibility bug. File an issue with the `replication` label.

---

## A. Code and software environment

| Item | Answer | Pointer |
|---|---|---|
| A1. Is the code publicly available? | YES | https://github.com/sam-siavoshian/Time-Model |
| A2. Is the code licensed for reuse? | YES | [LICENSE](LICENSE) (MIT). Synthetic data generators also MIT. Base model `Qwen/Qwen2.5-3B-Instruct` governed by Tongyi Qianwen license; not redistributed. |
| A3. Is a dependency manifest provided? | YES | [pyproject.toml](pyproject.toml) lines 1-18, [uv.lock](uv.lock) (full transitive lock) |
| A4. Is the language version pinned? | YES | `requires-python = ">=3.11,<3.14"` in [pyproject.toml:7](pyproject.toml). Tested on 3.11.15 (Mac mini M4) and 3.11.x (Spark GB10). |
| A5. Is the install procedure documented? | YES | [README.md:57-65](README.md) (`git clone && uv sync`). One command. |
| A6. Is an environment spec (OS / driver / CUDA) documented? | YES | Spark GB10 = Ubuntu aarch64 6.17, CUDA 13.0, NVIDIA driver 580.142. Mac mini M4 = macOS Darwin 25.x arm64. See `~/.claude/CLAUDE.md` "Spark" + "Mac mini" sections; mirrored in [README.md:245-253](README.md). |
| A7. Is the code self-contained (no hidden private deps)? | YES | All scripts in `model/`, `scripts/`, `data_gen/` are in-repo. Only external pulls are `Qwen/Qwen2.5-3B-Instruct` (HuggingFace, gated) and the Python deps in `pyproject.toml`. |
| A8. Is there a smoke test reachable without a GPU? | YES | `bash scripts/e2e_smoke.sh` (CPU, ~10 min on M-series; documented [README.md:103-108](README.md)). Generates 200 conversations, trains 20 steps, runs full 5-test eval. Numbers will be near-baseline; verifies the pipeline executes. |

## B. Data

| Item | Answer | Pointer |
|---|---|---|
| B1. Are all datasets used in the paper publicly available? | YES, but all are **synthetic and generated deterministically inside this repo**. Nothing to download. | [README.md:156-169](README.md), [data/README.md](data/README.md), [data/VERIFICATION.md](data/VERIFICATION.md). |
| B2. Is the data generation procedure documented? | YES | [model/qwen_time_data.py](model/qwen_time_data.py) (clock / silent-gap / phase generator). Three task families with explicit mix ratios. |
| B3. Is the data generation deterministic given a seed? | YES | `python -m model.qwen_time_data --n 18000 --seed <S> --mix 0.40,0.30,0.30 --out <path>`. Seed flows through Python `random.Random(seed)`. Deterministic across Python 3.11 patch versions (not guaranteed across minor versions). |
| B4. Are dataset hashes provided for the exact splits used in the paper? | YES | [data/VERIFICATION.md:7-26](data/VERIFICATION.md). SHA256 for all three v15 cross-seed training files: seed0 = `497b2392c785504b1a2be5ddd0f09838a99d90508c89012096bdb873d88d3d76`, seed1 = `6bbc71457fab392c0f4b4b6954efe95d047b2d02cbf76ae47997a8267bb492a4`, seed2 = `34db9c02bcdfe9c0ab651761a478b54408a2c2271b4d9b38fee166dda1406117`. Anchor v15 file (single-seed, seed=15): `09b8ee1e71b224a98cdabf0b7c225f6a94ce96a376393272fefb62f7803507d8`. |
| B5. Is dataset size documented? | YES | v15 training set = 18 000 conversations per seed, mix 0.40 clock / 0.30 silent-gap / 0.30 phase. Total per file ~6.5 MB JSONL. Full data zoo summary in [README.md:160-169](README.md): ~226 K examples, ~323 M tokens across all generators (most are for the abandoned IPCN tracks, archived in §D.21/§D.22). |
| B6. Are train / validation / test splits defined? | YES | Training data is `train_v15s_seed{0,1,2}_18k.jsonl`. Evaluation is a separate deterministic sampler inside `model/qwen_time_check.py` (5 tests, fresh τ per call from `numpy.random.default_rng(0)`). T1b OOD τ are drawn log-uniform in `[2 s, 14 d]`; genuine-OOD T1b in `[7 d, 28 d]`. T3 multi-week τ are fixed `(5.5 d, 12.5 d, 19.5 d, 26.5 d)`. |
| B7. Are schema and integrity checks on the data run? | YES | `data_gen/verify_data.py` runs structural checks on every generated dataset. Latest output in [data/VERIFICATION.md:30-189](data/VERIFICATION.md). All checks PASS. |
| B8. Is the data documented in a dataset card? | YES | [data/README.md](data/README.md). |
| B9. Are PII / consent / licensing of source data handled? | YES, N/A on source data because **all data is synthetic, generated from prompts written by the author**. No human subjects, no scraped content, no third-party IP. The optional Gutenberg slice (6 348 chunks) used in pre-pivot IPCN experiments is public-domain text; not used in the v15 paper claims. |

## C. Compute

| Item | Answer | Pointer |
|---|---|---|
| C1. Is the hardware used in the paper documented? | YES | Training: 1× NVIDIA Grace-Blackwell GB10 (DGX Spark prototype, sm_120, CUDA 13.0, 121 GB unified memory, 20 ARM cores). Data prep + CPU eval: 1× Apple Mac mini M4 (16 GB unified, MPS). See [README.md:250-251](README.md). |
| C2. Are wall-clock training times per result reported? | YES | Per-seed v15 training = ~45 min on GB10 (18 000 steps). Cross-seed total = 3 × ~45 min ≈ 2.25 GPU-hours. Falsification battery on a checkpoint = ~10 min. Linear probe = ~30 min. Pressure v2 = ~15 min. Full table in [README.md:77-97](README.md). |
| C3. Is total compute for the paper reported? | YES | See "Computational budget summary" at the bottom of this file. Total ≈ 10.5 GPU-hours on GB10 across all v15-era runs that appear in §24.7. |
| C4. Is the per-run memory footprint reported? | YES | Peak GPU memory at chunk_length=512 ~ 30 GB on GB10 (LoRA + chrono encoder + per-layer FiLM projectors + Qwen 2.5 3B in bf16). 14B base OOMs the 128 GB pool ([PAPER.md §24.6.5](PAPER.md)). LoRA-only ablation has identical footprint (α just frozen, not removed). |
| C5. Is a single-GPU recipe provided? | YES | Every command in [README.md:77-97](README.md) runs on a single GPU. No model parallelism or sharding required for 3B base. |
| C6. Are CPU-only paths documented? | YES | `scripts/e2e_smoke.sh` runs the full pipeline on CPU. `model/qwen_time_check.py` and `model/qwen_time_probe.py` accept `--device cpu` (slow but works). |

## D. Hyperparameters

| Item | Answer | Pointer |
|---|---|---|
| D1. Are all hyperparameters disclosed? | YES | Defaults: [model/qwen_time_train.py:80-110](model/qwen_time_train.py) (`--steps 8000`, `--lr 1e-4`, `--chunk-length 512`, `--seed 0`, `--base Qwen/Qwen2.5-3B-Instruct`, `--injection-type film`). Architecture defaults: [model/qwen_time.py:44-58](model/qwen_time.py) (`lora_rank=8`, `lora_targets=("q_proj","k_proj","v_proj","o_proj")`, `lora_lm_head=True`, `inject_layers=()` = inject at every layer, `injection_type="film"`). |
| D2. Are paper-run overrides documented? | YES | v15 overrides in [scripts/run_v15.sh](scripts/run_v15.sh): `--steps 18000`, `--lr 1e-4`, `--chunk-length 512`, `--timescales 2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800` (15 scales including day + week, replacing the 13-scale default). Same overrides in [scripts/run_v15_seeds.sh](scripts/run_v15_seeds.sh) per seed. |
| D3. Are training-data mix ratios disclosed? | YES | `--mix 0.40,0.30,0.30` (clock / silent-gap / phase) for v15. Within-phase 50/50 weekday/weekend balance is default since v14 in `gen_phase_conversation` ([model/qwen_time_data.py](model/qwen_time_data.py)). |
| D4. Is the optimizer reported with all settings? | YES | AdamW (PyTorch default β, ε), `lr=1e-4`, `weight_decay=0.01`, gradient clipping `max_norm=1.0`, `set_to_none=True` zero-grad. See [model/qwen_time_train.py:163-170](model/qwen_time_train.py). |
| D5. Are LoRA / adapter parameters disclosed? | YES | Rank 8 (default `lora_rank=8` in [model/qwen_time.py:49](model/qwen_time.py)), α scaling 16.0 (so `scaling = alpha/rank = 2.0`), targets `q_proj, k_proj, v_proj, o_proj` + `lm_head`. The README mentions "rank 16" in the method narrative; that is a historical artifact from an earlier draft. The code default and the paper-headline checkpoints use rank 8. |
| D6. Is the chronometric encoder spec disclosed? | YES | 27-dim output = 13 sin + 13 cos + 1 log1p(τ) for the default 13-scale set; 31-dim output = 15 sin + 15 cos + 1 log1p(τ) for v15's 15-scale set (with day + week added). Definition in [model/qwen_time.py:61-77](model/qwen_time.py). |
| D7. Is the FiLM injection mathematically defined? | YES | `h' = h + α · (γ · h + β)` with α initialized to 0 and γ-bias initialized to 1 (DiT AdaLN-Zero init). [model/qwen_time.py:101-165](model/qwen_time.py). γ, β are linear projections of the 27-dim chrono vector. |
| D8. Is hyperparameter selection methodology disclosed? | YES, **no hyperparameter sweep was run for the v15 paper headline**. All v15 settings were carried over from v14, with three deliberate changes documented in [PAPER.md §24.7.2](PAPER.md): 18 K records (vs 6 K), 18 K steps (vs 12 K), 15-scale encoder (vs 13). The 5-test thresholds were locked in [PREREGISTRATION.md](PREREGISTRATION.md) before v15 trained. Each prior version (v11-v14) was a single deliberate change, recorded in [PAPER.md §24.6.1-§24.6.3](PAPER.md). |

## E. Random seeds and determinism

| Item | Answer | Pointer |
|---|---|---|
| E1. Are seeds set for every randomness source? | YES, where it matters. `torch.manual_seed(args.seed)` at the top of `train_step` setup ([model/qwen_time_train.py:112](model/qwen_time_train.py)). Data shuffling uses `random.Random(seed)` in `stream_records`. Evaluation samplers use `numpy.random.default_rng(0)`. CUDA non-determinism is NOT explicitly disabled (`torch.use_deterministic_algorithms(True)` is not set); see E3. |
| E2. Which results are reported with cross-seed variance? | YES, every v15-era headline number is cross-seed (n=3, seeds 0, 1, 2). T1, T1b, T2, T3, T4 all reported as mean ± std in [PAPER.md §24.7.5](PAPER.md) table. LoRA-only baseline also cross-seed n=3 ([PAPER.md §24.7.8 cross-seed update](PAPER.md)). The α-flip causal intervention (r = −0.9998) is single-seed (v15 seed 0); the half-layer-flip control results in [PAPER.md §24.7.9](PAPER.md) are also seed-0 only. This is flagged as a limitation. |
| E3. Are deterministic vs nondeterministic paths called out? | YES. **Deterministic:** data generation (Python `random`), training data shuffling, evaluation prompt selection (numpy RNG with seed 0), all greedy decoding. **Nondeterministic on CUDA:** training itself (no `torch.use_deterministic_algorithms(True)`, no `CUBLAS_WORKSPACE_CONFIG`). The [README.md:249](README.md) discloses: "Expect ±0.02 noise on Pearson r across hardware." Cross-seed std (e.g. T1 = 0.961 ± 0.035) absorbs this. |
| E4. Is the seed used for the headline checkpoints documented? | YES | v15 cross-seed: seeds 0, 1, 2 (filename suffix encodes the seed). The anchor v15 single-seed run that hit T1 = 0.9997 used seed 15 (overall best seed of all v15-era runs, used for the architecture demo in [PAPER.md §24.7.2](PAPER.md) but NOT for the cross-seed mean ± std table). |

## F. Negative results and failure modes

| Item | Answer | Pointer |
|---|---|---|
| F1. Are negative results reported? | YES, prominently. Three large negatives: (a) the original IPCN memory routing failed across 9 variants ([PAPER.md Appendix D §D.21, §D.22](PAPER.md)); (b) the OOD-behavioral-transfer claim is **retracted** ([PAPER.md §24.7.3](PAPER.md)); (c) genuine OOD on τ ∈ [7 d, 28 d] fails with r = −0.20 ([PAPER.md §24.7.1](PAPER.md)). All three appear in the conclusion ([PAPER.md §25, §26.2](PAPER.md)). |
| F2. Are abandoned hypotheses documented? | YES | The 7 pre-registered hypotheses in [PREREGISTRATION.md §2](PREREGISTRATION.md) include H1-H4, H5, H7 (memory-routing). All are abandoned per [PAPER.md §22 / Appendix D §D.22](PAPER.md). Only H6 (chronometric substrate) survives and is the paper claim. |
| F3. Are failed architectural variants reported? | YES | Two architectural ablations in [PAPER.md §24.7.10](PAPER.md): additive injection (no FiLM) → all 5 tests collapse to 0; L0-only injection (single layer instead of every layer) → matches v15 on 4/5 tests, loses T1b precision. The additive failure is mechanistically explained (α-gradient trap at init). |
| F4. Are seed-level failures within an experiment reported? | YES | T3 cross-seed: seed 1 mode-collapses; seeds 0 and 2 pass. Reported as "2 of 3 seeds pass" in [PAPER.md §24.7.5](PAPER.md) rather than as a Gaussian mean ± std (the underlying outcome is binary). |
| F5. Are failed eval metrics reported even when other metrics pass? | YES | The original single-seed v15 T4 = 0.016 (below threshold 0.05) is reported in [PAPER.md §24.7.2](PAPER.md) alongside the 4 passes. Cross-seed T4 averaged 0.18 (passes), and that retraction-of-a-retraction is documented in [PAPER.md §24.7.5](PAPER.md) item 4. |
| F6. Are pathological evaluation regimes reported? | YES | Probe v5 with prediction clamping ([PAPER.md §24.7.14](PAPER.md)) shows the original probe R² = +0.43 collapses to −2.42 on a clamped within-distribution split. The "R² = −143 chrono-off" number is partially a ridge-solver pathology on degenerate variance, not a clean signal-destruction measurement. This is documented as a real limitation. |

## G. Reproduction instructions

| Item | Answer | Pointer |
|---|---|---|
| G1. Is there a one-command path to the headline numbers? | YES | `bash scripts/run_v15_seeds.sh && uv run python scripts/aggregate_seeds.py` reproduces the n=3 cross-seed mean ± std table. ~2.25 GPU-hours on GB10. [README.md:79](README.md). |
| G2. Is the eval harness separate from the training harness? | YES | Training: `model/qwen_time_train.py`. Eval: `model/qwen_time_check.py` (5-test), `model/qwen_time_falsify.py` (causal interventions), `model/qwen_time_pressure_v2.py` (n=30 bootstrap CI), `model/qwen_time_probe.py` (linear probe), `model/qwen_time_check_genuine_ood.py` (truly held-out T1b + multi-week T3), `model/qwen_time_extra_controls.py` (paraphrase + half-flip + teacher-forced T4), `model/qwen_time_alpha_norms.py` (per-layer α dump + targeted flip), `model/qwen_time_t2t3_sampling.py` (temp=0.7 effective-n fix). |
| G3. Is figure regeneration scripted? | YES | `uv run python scripts/make_figures.py && uv run python scripts/make_fig5.py && uv run python scripts/make_fig6.py`. CPU. Reads from `reports/*.json`. [README.md:95](README.md). |
| G4. Is the LoRA-only baseline reproducible from one command? | YES | `bash scripts/run_baseline_lora.sh` (single seed) or `bash scripts/run_lora_seeds.sh` (seeds 1 and 2 to complement seed 0). [README.md:80-81](README.md). |
| G5. Are architectural ablations scripted? | YES | `bash scripts/run_ablation_l0_only.sh` and `bash scripts/run_ablation_additive.sh`. [README.md:85-86](README.md). |
| G6. Is data regeneration scripted? | YES | `uv run python -m model.qwen_time_data --n 18000 --seed <SEED> --mix 0.40,0.30,0.30 --out <path>`. Verify via `sha256sum` against the table in [data/VERIFICATION.md:13-16](data/VERIFICATION.md). |
| G7. Is there a tolerance bound for "reproduced"? | YES | "Numbers in the result table will match the paper within ±0.02 on Pearson r." [README.md:98](README.md). For KL metrics, ±5% ([README.md:252](README.md)). Wider drift means a real difference, not noise. |

## H. Limitations

| Item | Answer | Pointer |
|---|---|---|
| H1. Are limitations explicitly listed? | YES | [PAPER.md §24.7.14](PAPER.md) (probe limit), [PAPER.md §24.7.6](PAPER.md) (effective-n disclosure), [PAPER.md §25.1](PAPER.md) (5 open questions including scale beyond 3B, deep-layer mechanism, persistence under no-input, other modalities), [PAPER.md §26](PAPER.md) (claims-that-do-not-survive table). |
| H2. Is the OOD-extrapolation limit characterized? | YES | Sinusoidal encoder cannot extrapolate beyond largest training timescale (604 800 s = 7 d). Genuine OOD on τ ∈ [7 d, 28 d] gives r = −0.20, log-MAE = 0.49. Phase encoding has a ~14-day horizon. [PAPER.md §24.7.1](PAPER.md). |
| H3. Is the effective-n issue characterized? | YES | [PAPER.md §24.7.6](PAPER.md): T1 reported n=64 → effective n=8 unique τ under greedy; T3 reported n=20 → effective n=2 unique prompts; α-flip reported n=32 → effective n=3 parsed τ. Rerun under temp=0.7 sampling for T2/T3 gives genuine n=30 ([PAPER.md §24.7.13](PAPER.md)). |
| H4. Is the seed-fragility on T3 characterized? | YES | T3 is a binary per-seed outcome. 2/3 seeds pass; seed 1 mode-collapses to a single response template regardless of τ. [PAPER.md §24.7.5](PAPER.md). |
| H5. Is the per-layer-flip nuance characterized? | YES | The "single coherent scalar dial" framing is too strong. Half-layer-flip (17 of 35 layers) gives r = +0.78 or r = −0.93 depending on which 17. Top-8-dominant-layer flip zeros the signal; bottom-8 flip preserves it. The chrono pathway is a weighted layer vote with a dominant mid-deep subset (L19-L28). [PAPER.md §24.7.9, §24.7.11](PAPER.md). |
| H6. Is the scale limit documented? | YES | 14B OOMs on a 128 GB GB10. 7B partially works but T1 in-distribution dips to 0.747 at 12 K steps. Only 3B is fully validated. [PAPER.md §24.6.4, §24.6.5](PAPER.md). |
| H7. Is the cross-model-base limit documented? | YES | Only Qwen 2.5 3B-Instruct tested. Other bases listed as future work in [PAPER.md §25.1](PAPER.md) and [README.md:203](README.md). |

## I. Pre-registration

| Item | Answer | Pointer |
|---|---|---|
| I1. Was the experiment pre-registered? | YES (informally, without OSF stamp). Locked 2026-05-12. [PREREGISTRATION.md](PREREGISTRATION.md). All thresholds for the 5 falsifiable tests and the 3-experiment disproof battery were committed to the repo before v11 trained. Final post-empirical version moved to body as [PAPER.md §23.9 + §D.16](PAPER.md). |
| I2. Were thresholds locked before training? | YES | T1 ≥ 0.8, T1b r ≥ 0.7 + log-MAE < 0.5, T2 Δ ≥ 0.5, T3 signal ≥ 0.3, T4 KL ≥ 0.05. Locked in [PREREGISTRATION.md §2](PREREGISTRATION.md) and unchanged through every training run from v11 to v15. |
| I3. Were any pre-registered hypotheses amended after seeing data? | YES, transparently. The original 7 hypotheses (H1-H7) targeted memory routing. H1-H5 + H7 were abandoned after the Track B null result ([PAPER.md §D.22](PAPER.md)). H6 (chronometric substrate) survived and became the paper claim. The pivot is documented as Section D.22.3. No threshold was loosened after seeing data; the architectural claim was narrowed. |
| I4. Were rigor reruns pre-registered? | YES (one round). The §24.7 reviewer-rigor audit pre-registered three attacks before running the v2 reruns: P2 with n=30 + max=256 + bootstrap CI, genuine-OOD T1b on [7 d, 28 d], T3 multi-week. Pre-registered predictions and decision rules in [PAPER.md §24.6.6b](PAPER.md) BEFORE the reruns ran. The OOD retraction in §24.7.3 honored the pre-registered "if CI crosses zero, claim is retracted" rule. |
| I5. Is the pre-registration version-controlled? | YES | [PREREGISTRATION.md](PREREGISTRATION.md) is in git; `git log PREREGISTRATION.md` shows the lock date (2026-05-12) precedes the first v11 training run by several days. |

## J. Checkpoints and artifacts

| Item | Answer | Pointer |
|---|---|---|
| J1. Are trained checkpoints released? | YES | GitHub Release [`v15.0`](https://github.com/sam-siavoshian/Time-Model/releases/tag/v15.0). Three checkpoints, one per seed, ~38 MB each (trainable parameters only: LoRA + chrono encoder + per-layer FiLM projectors). |
| J2. Are checkpoint hashes provided? | YES | seed 0 SHA256 = `2ab64f3f837f58ca726297bad61e1c606fac03d7567884775bc6601cc429ecef`; seed 1 = `51bc2425cd406ed0ec433405bfef745f7e13a7ca1ae4e154eddc5b997980ef58`; seed 2 = `d718baf88b509d76c371602f27fec8d703b5a556888a9ceea613ffdfe41ce7c0`. [README.md:136-140](README.md). |
| J3. Is checkpoint load instructions documented? | YES | [README.md:142-150](README.md). One command: `curl -L -o ckpt.pt <URL> && sha256sum ckpt.pt && uv run python -m model.qwen_time_check --checkpoint ckpt.pt --base Qwen/Qwen2.5-3B-Instruct --timescales 2,4,8,...,604800 --out reports/recall.json`. Base model is fetched from HuggingFace on first run; user must accept the Qwen 2.5 license. |
| J4. Are intermediate JSON reports released? | YES | All `reports/*.json` are in-repo (committed). Includes per-seed `*_recall.json`, falsify, pressure v2, probe, paraphrase + half-flip + teacher-forced T4, α-norm dump, T2/T3 sampling rerun, ablation runs. ~50 JSON files. |
| J5. Are figures reproducible from the released JSONs? | YES | `scripts/make_figures.py`, `make_fig5.py`, `make_fig6.py` read only `reports/*.json` and write to `figures/`. No CUDA required. |

## K. Documentation and reporting

| Item | Answer | Pointer |
|---|---|---|
| K1. Is there a single entry-point README? | YES | [README.md](README.md). |
| K2. Is the paper preprint in the repo? | YES | [PAPER.md](PAPER.md). 2 644 lines, ~22 K words. |
| K3. Are claims and evidence cross-linked? | YES | Every claim in [PAPER.md §26.1](PAPER.md) table cites the supporting subsection and JSON report. |
| K4. Are retractions called out in the paper, the README, and the abstract? | YES | OOD-transfer retraction documented in [PAPER.md Abstract](PAPER.md) (item 2), [README.md:121-122, 50](README.md), and [PAPER.md §24.7.3, §25, §26.2](PAPER.md). |
| K5. Is there a machine-readable citation? | YES | [CITATION.cff](CITATION.cff) renders the "Cite this repository" button on GitHub. |

---

## Computational budget summary

All v15-era compute used a single NVIDIA Grace-Blackwell GB10 (DGX Spark prototype, 128 GB unified, sm_120, CUDA 13.0). Mac mini M4 used only for data generation and CPU eval smoke. No rented cloud, no multi-GPU.

| Experiment | GPU-hours | Hardware | Notes |
|---|---|---|---|
| v15 single-anchor training (seed 15) | ~0.75 | GB10 | The architecture-demo checkpoint that hit T1=0.9997. Not used for the cross-seed mean. |
| v15 cross-seed training (seeds 0, 1, 2) | ~2.25 | GB10 | 3 × ~45 min. Source of the paper's headline mean ± std numbers. |
| LoRA-only baseline (3 seeds, --freeze-alpha) | ~2.25 | GB10 | 3 × ~45 min. Source of the [PAPER.md §24.7.8](PAPER.md) all-zero collapse. |
| L0-only ablation (seed 0) | ~0.75 | GB10 | [PAPER.md §24.7.10](PAPER.md). |
| Additive (no-FiLM) ablation (seed 0) | ~0.75 | GB10 | [PAPER.md §24.7.10](PAPER.md). |
| 7B scale test | ~1.0 | GB10 | [PAPER.md §24.6.4](PAPER.md). |
| 14B scale attempt | ~0.5 | GB10 | OOM kill at step 1146 of 12 000; partial wallclock counted. [PAPER.md §24.6.5](PAPER.md). |
| Disproof battery on v11 anchor (falsify + pressure + probe) | ~1.0 | GB10 | [PAPER.md §24.1-§24.3](PAPER.md). |
| Pressure v2 (n=30, max_new=256, bootstrap CI) | ~0.25 | GB10 | [PAPER.md §24.7.3](PAPER.md). Single run on v15 ckpt. |
| Genuine OOD + multi-week T3 on v14 and v15 ckpts | ~0.4 | GB10 | [PAPER.md §24.7.1](PAPER.md). |
| Extra controls (paraphrase + half-flip + teacher-forced T4) | ~0.25 | GB10 | [PAPER.md §24.7.9, §24.7.12](PAPER.md). |
| α-norm dump + targeted flip | ~0.1 | GB10 | [PAPER.md §24.7.11](PAPER.md). |
| T2/T3 sampling rerun (temp=0.7) | ~0.15 | GB10 | [PAPER.md §24.7.13](PAPER.md). |
| Linear probe v5 (clamped) | ~0.5 | GB10 | [PAPER.md §24.7.14](PAPER.md). |
| Data generation (all 6 v15 datasets + verification) | ~0.1 CPU-hours | Mac mini M4 | Counted separately because no GPU is involved. |
| **Total GPU-hours** | **~10.9** | GB10 | All in-repo. |
| **Total CPU-hours** | **<1** | Mac mini M4 | Data prep + smoke. |

**Pre-pivot IPCN compute** (Track A from-scratch + Track B 9 variants of Qwen + memory routing) consumed an additional ~80 GPU-hours over three weeks on the GB10. None of that compute supports the v15 paper claims; it is preserved in [PAPER.md Appendix D §D.21, §D.22](PAPER.md) for honesty about how the work evolved. The full repo is reproducible from a fresh checkout in ~11 GPU-hours of GB10 time.

For a reviewer who wants to verify the headline claim only (the n=3 cross-seed table in [PAPER.md §24.7.5](PAPER.md)), the minimum reproduction cost is **2.25 GPU-hours** on a single H100 / GB10 / A100 80 GB.
