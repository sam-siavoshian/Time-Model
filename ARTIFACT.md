# Artifact Map

This is the repo-level source-of-truth map for current paper claims and result
files. Treat `paper/main.tex` as the canonical manuscript and
`reports/current/manifest.json` as the canonical machine-readable evidence
index. The manifest is generated from `reports/current/claims.json` and checked
with `python3 scripts/generate_current_manifest.py --check`.

## Current Claims

| Claim ID | Status | Paper location | Canonical evidence | Generator / command | Caveat |
|---|---|---|---|---|---|
| `mechanistic_probe` | current | `paper/main.tex` Results VI-A | `reports/probe_within_dist_v15s_seed0.json`; `reports/probe_per_layer_v15s_s0.json` | `model/qwen_time_probe*.py` | Probe is in-distribution over the trained tau range unless explicitly marked OOD. |
| `clock_heldout_probe` | current | `paper/main.tex` Results VI-A | `reports/probe_per_layer_clock_heldout_s{0,1,2}.json` | `scripts/run_probe_clock_heldout.sh`; `scripts/build_probe_paper_update.py` | Supports channel transmission, not behavior without task training. |
| `v15_behavior` | current | `paper/main.tex` Results VI-C | `reports/v15_cross_seed_aggregate.json` and raw v15 seed reports | `scripts/run_v15_seeds.sh`; `scripts/aggregate_seeds.py` | T1/T1b are supervised wiring checks. |
| `ablation_controls` | current | `paper/main.tex` Results VI-B/VI-D | `reports/chrono_only_s*.json`, `reports/qwen_time_lora_only_*`, `reports/additive_nonzero_beta_s*.json`, `reports/ia3_*` | ablation launchers under `scripts/` | Some older additive and sign-flip artifacts are historical controls, not headline claims. |
| `tau_sessions` | current negative | `paper/main.tex` Results VI-E | `reports/ext_bench_vanilla.json`; `reports/ext_bench_prompt.json`; `reports/ext_bench_ci.json` | `scripts/run_external_bench.sh` | Prompt beats CI on composite and adaptive length correlation in current reports. |
| `tpdr_v2` | current with caveat | `paper/main.tex` Results VI-F | `reports/tpdr_v2_headline.json`; `reports/tpdr_crossseed/` | `eval/tpdr/analyze_v2.py`; `scripts/build_tpdr_paper_update.py` | Headline seed pair confirms; cross-seed behavior is unstable. |
| `tps` | current negative | `paper/_6g_tps.tex`; `paper/main.tex` | `reports/tps/headline.json`; `reports/tps/baselines.json` | `eval/tps/run_tps.py`; `eval/tps/analyze.py`; `scripts/build_tps_paper_update.py` | CI v15s does not beat vanilla hidden-only and monotonicity is negative. |
| `track_b_policy` | current with caveat | `paper/_6g_tps.tex`; `paper/main.tex` | `reports/track_b_policy_headline.json` | `scripts/run_track_b_policy.sh` | Separate policy-trained Track B checkpoints show hidden-time action control, but held-out `market_data` transfer remains weak. |

## Archive Boundaries

Archived files remain tracked for provenance but are not current evidence:

- `docs/history/`: old lab notebook and handoff docs.
- `reports/archive/ipcn_memory/`: pre-pivot IPCN/memory-routing artifacts.
- `reports/archive/model_versions_v10_v14/`: pre-v15 model-version reports.
- `reports/archive/tpdr_v1/`: TPDR v1 outputs superseded by TPDR v2.
- `reports/archive/superseded_prompt_baseline/`: contaminated or superseded prompt baselines.

Each archive directory has a `MANIFEST.md` explaining the move and successor.
