# Script Entrypoints

Supported shell runners live directly in this folder. They all use
`scripts/lib/run_context.sh` and require `--run-id <id>`, `--resume-run-id <id>`,
or `RUN_ID=<id>`.

New outputs must be written under:

- `runs/<run_id>/data/`
- `runs/<run_id>/logs/`
- `runs/<run_id>/checkpoints/`
- `runs/<run_id>/reports/`

Supported track runners:

- `run_track_a_mechanistic.sh`: Track A alias for the canonical v15
  CLOCK/SILENT-GAP/PHASE mechanistic CI run.
- `run_v15_seeds.sh`: underlying Track A cross-seed runner preserved for
  compatibility.
- `run_track_b_policy.sh`: Track B TPS policy-training runner for
  REUSE/REFRESH/ASK/SUMMARIZE forced-choice labels.

Historical launchers live in `scripts/legacy/`. They are retained for provenance
and may still be useful when reconstructing old runs, but they are not supported
reproduction entrypoints and should not be referenced as current generators.
