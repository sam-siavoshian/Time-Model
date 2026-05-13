# IPCN — Path Conventions

Canonical filesystem layout. Used by all training, eval, monitoring tools.

---

## Repository root

`/Users/samsiavoshian/Desktop/Coding Stuff/Time-Model/` (local laptop)
`~/Desktop/Time-Model/` (DGX Spark, after `rsync`)
GitHub: https://github.com/sam-siavoshian/AGI

---

## Generated artifacts

### Datasets (gitignored, regenerable)
```
data/
  latent_world/        train_{1k,2k,4k,8k}.jsonl, test_{16k,32k,64k,128k}.jsonl, valid_{1k,4k,16k}.jsonl
  ambiguity/           train.jsonl, valid.jsonl
  consolidation/       ladder_train.jsonl
  chronometric_pairs/  pairs.jsonl
  contradiction_pairs/ pairs.jsonl
  real_text/           gutenberg.jsonl
  tokenized/           <type>/<split>.{tokens.bin,boundaries.npy,meta.json}
  README.md            dataset card
  VALIDATION.md        stats
  VERIFICATION.md      schema correctness
```

### Checkpoints
```
checkpoints/
  phase0_sanity.pt              final Phase 0 checkpoint
  phase1_pfc_consolidation.pt   final Phase 1 checkpoint
  phase2_early_core.pt          final Phase 2 checkpoint
  phase3_mixed_lm.pt            final Phase 3 checkpoint
  phase0_sanity_step{N}.pt      intermediate Phase 0 checkpoints (every 10k by default)
  phase1_pfc_consolidation_step{N}.pt
  phase2_early_core_step{N}.pt
  phase3_mixed_lm_step{N}.pt
  ablation_v1/A{0..6}.pt        ablation matrix variants
  keep/                         user-curated subdir (cleanup script ignores)
```

### Logs
```
logs/
  phase0_sanity.jsonl           Phase 0 training log (one JSON record per step + consolidation events)
  phase1_pfc_consolidation.jsonl
  phase2_early_core.jsonl
  phase3_mixed_lm.jsonl
```

### Reports
```
reports/
  preflight.md                  scripts/preflight.py output
  untrained_baseline_v2.{md,json}
  e2e_smoke/00_baseline.{md,json}
  e2e_smoke/01_phase0.{md,json}
  e2e_smoke/02_phase1.{md,json}
  alerts.jsonl                  monitor/safety alert events (append-only)
  phase{N}.{md,json}            per-phase post-training eval (created by eval_all)
  ablation_v1/summary.md
```

---

## Convention rules

1. Final checkpoint per phase: `checkpoints/<phase_name>.pt`. Phase names match `Phase.value.name`.
2. Intermediate checkpoints: `checkpoints/<phase_name>_step<N>.pt`. Auto-cleaned by `scripts/cleanup.sh` (skip if under `keep/`).
3. Per-phase log: `logs/<phase_name>.jsonl`. One JSON record per training step, plus separate records with `"event": "consolidation"`.
4. Per-phase post-training eval: `reports/<phase_name>.md` and `.json`.
5. Alerts: `reports/alerts.jsonl` append-only, line-delimited JSON.
6. Ablation outputs: `reports/ablation_<tag>/` and `checkpoints/ablation_<tag>/`.

---

## File-size estimates

| Item | Size |
|---|---|
| Final phase checkpoint | ~410 MB |
| Intermediate phase checkpoint | ~1.2 GB (includes optimizer state) |
| Training log (per phase, 100k steps) | ~30 MB |
| Tokenized binary cache (total) | 620 MB |
| Raw JSONL datasets (total) | 3 GB |
| All 4 phase checkpoints + 30 intermediates | ~40 GB |

Total disk requirement on Spark for full Phase 0-3: **~45 GB** with intermediates (Spark has 228 GB).

---

## Sync rules (laptop -> Spark)

Push (laptop -> Spark) before training:
- All of `data/` (3 GB raw + 620 MB tokenized)
- All of `model/` and `data_gen/` and `scripts/`
- `pyproject.toml`, `uv.lock`, `.python-version`
- All docs

Skip: `.venv/`, `__pycache__/`, `logs/`, `checkpoints/`, `reports/`

Pull (Spark -> laptop) after training:
- `checkpoints/` (final + 1-2 intermediates per phase if want)
- `logs/`
- `reports/`
