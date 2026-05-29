# Chronometric Injection

Chronometric Injection (CI) adds a continuous elapsed-time side channel to a
frozen Qwen 2.5 3B-Instruct model. The main paper studies whether elapsed
seconds can be made mechanically recoverable inside the residual stream, whether
that path is causal under gate interventions, and where the resulting behavior
does or does not improve over prompt-injected timestamps.

Canonical paper source: [`paper/main.tex`](paper/main.tex)

Artifact manifest: [`ARTIFACT.md`](ARTIFACT.md)

Canonical report bundle: [`reports/current`](reports/current)

Training-track map: [`docs/TRAINING_TRACKS.md`](docs/TRAINING_TRACKS.md)

## Scope

CI is a residual-stream input-channel experiment, not a claim of subjective time
perception. The model does not autonomously track time between forward passes;
it conditions on an externally supplied elapsed-time scalar.

The defensible claim is narrow:

- A 31-dimensional chronometric vector `χ(τ)` encodes elapsed seconds with 15
  sinusoidal timescales plus `log(1+τ)`.
- The channel is injected into 35 of 36 Qwen decoder layers with
  AdaLN-Zero-style FiLM:
  `h' = h + α * (γ(χ) * h + β(χ))`.
- The frozen base remains unchanged; trainable parameters are confined to
  chrono projectors/gates and the PEFT surface used by the experiment.
- Rank-8 LoRA is applied to attention projections plus `lm_head` in the main
  configuration.
- Prompt timestamps are strong baselines and beat CI on several direct recall
  settings. CI should not be described as a universal replacement for prompt
  injection.

## Install

```bash
git clone https://github.com/sam-siavoshian/Time-Model
cd Time-Model
uv sync
```

Training/evaluation against the Qwen base requires Hugging Face access to
`Qwen/Qwen2.5-3B-Instruct`.

## Reproduce

Track A mechanistic CI, the current paper evidence track:

```bash
bash scripts/run_track_a_mechanistic.sh --run-id ci_track_a_v15s
uv run python scripts/aggregate_seeds.py \
  --inputs \
  runs/ci_track_a_v15s/reports/track_a/ci_track_a_v15s_seed0_recall.json \
  runs/ci_track_a_v15s/reports/track_a/ci_track_a_v15s_seed1_recall.json \
  runs/ci_track_a_v15s/reports/track_a/ci_track_a_v15s_seed2_recall.json \
  --out runs/ci_track_a_v15s/reports/track_a/ci_track_a_v15s_aggregate.json
```

Track B policy CI, the experimental TPS policy-training track:

```bash
bash scripts/run_track_b_policy.sh --run-id ci_track_b_policy
```

Run a released/checkpointed CI model through the core evaluator:

```bash
uv run python -m model.qwen_time_check \
  --checkpoint ckpt.pt \
  --base Qwen/Qwen2.5-3B-Instruct \
  --timescales 2,4,8,16,32,64,128,256,512,1024,4096,16384,65536,86400,604800 \
  --out runs/manual_eval/reports/recall.json
```

## Current Evidence Files

Core v15 cross-seed:

- `reports/v15_cross_seed_aggregate.json`
- `reports/qwen_time_v15s_20260523_141410_seed0_recall.json`
- `reports/qwen_time_v15s_20260523_141410_seed1_recall.json`
- `reports/qwen_time_v15s_20260523_141410_seed2_recall.json`

Mechanistic and causal controls:

- `reports/probe_per_layer_v15s_s0.json`
- `reports/probe_per_layer_clock_heldout_s0.json`
- `reports/probe_per_layer_clock_heldout_s1.json`
- `reports/probe_per_layer_clock_heldout_s2.json`
- `reports/alpha_norms_cross_seed.json`
- `reports/alpha_flip_permutation_100.json`

Baselines and ablations:

- `reports/prompt_baseline_injected_s0_recall.json`
- `reports/prompt_baseline_injected_s1_recall.json`
- `reports/prompt_baseline_injected_s2_recall.json`
- `reports/chrono_only_s0_recall.json`
- `reports/chrono_only_s1_recall.json`
- `reports/chrono_only_s2_recall.json`
- `reports/additive_nonzero_beta_s0_recall.json`
- `reports/additive_nonzero_beta_s1_recall.json`
- `reports/additive_nonzero_beta_s2_recall.json`
- `reports/ia3_with_chrono_s0_recall.json`
- `reports/ia3_with_chrono_s1_recall.json`
- `reports/ia3_with_chrono_s2_recall.json`
- `reports/ia3_only_s0_recall.json`
- `reports/ia3_only_s1_recall.json`
- `reports/ia3_only_s2_recall.json`

External/behavioral benchmarks:

- `reports/ext_bench_ci.json`
- `reports/ext_bench_prompt.json`
- `reports/ext_bench_vanilla.json`
- `reports/tpdr_v2_headline.json`
- `reports/tps/headline.json`

Legacy IPCN and memory-routing reports are preserved under `reports/archive/`
for provenance, but they are not current CI evidence.

## Training Tracks

- Track A: `CLOCK`, `SILENT-GAP`, and `PHASE`; supports the current paper's
  mechanistic residual-stream elapsed-time channel claim.
- Track B: TPS forced-choice policy labels `REUSE`, `REFRESH`, `ASK`, and
  `SUMMARIZE`; experimental until policy-trained results are produced.
- Track C: combined Track A+B training is deliberately absent for now. It should
  only be added after Track B works independently.

## Checkpoints

The v15 cross-seed trainable checkpoints are released under GitHub Releases
`v15.0`. They contain adapter/chrono weights only and must be loaded on top of
the frozen Qwen base.

| Seed | File | SHA256 |
|---|---|---|
| 0 | `qwen_time_v15s_20260523_141410_seed0.pt` | `2ab64f3f837f58ca726297bad61e1c606fac03d7567884775bc6601cc429ecef` |
| 1 | `qwen_time_v15s_20260523_141410_seed1.pt` | `51bc2425cd406ed0ec433405bfef745f7e13a7ca1ae4e154eddc5b997980ef58` |
| 2 | `qwen_time_v15s_20260523_141410_seed2.pt` | `d718baf88b509d76c371602f27fec8d703b5a556888a9ceea613ffdfe41ce7c0` |

## What This Does Not Claim

- It does not show subjective or autonomous time perception.
- It does not show that residual injection is generally better than writing
  timestamps into the prompt.
- It does not show zero-shot policy transfer on Temporal Policy Switching; TPS
  is negative for CI v15s in the current paper.
- It does not depend on the abandoned IPCN-era memory-routing claim.

## Citation

```bibtex
@misc{siavoshian2026chronometric,
  title  = {Chronometric Injection: Causal Elapsed-Time Conditioning in a Frozen Language Model},
  author = {Sam Siavoshian and Omar Ramadan},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/sam-siavoshian/Time-Model}
}
```

Code is MIT licensed. Qwen base weights are governed by the upstream Qwen
license and are not redistributed here.
