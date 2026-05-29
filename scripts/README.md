# Script Entrypoints

Supported shell runners live directly in this folder. They all use
`scripts/lib/run_context.sh` and require `--run-id <id>`, `--resume-run-id <id>`,
or `RUN_ID=<id>`.

New outputs must be written under:

- `runs/<run_id>/data/`
- `runs/<run_id>/logs/`
- `runs/<run_id>/checkpoints/`
- `runs/<run_id>/reports/`

Historical launchers live in `scripts/legacy/`. They are retained for provenance
and may still be useful when reconstructing old runs, but they are not supported
reproduction entrypoints and should not be referenced as current generators.
