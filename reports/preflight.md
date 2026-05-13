# IPCN Preflight Report

- Total checks: 13
- Passed: 12
- Failed: 1

| Check | Status | Detail |
|---|---|---|
| python deps | PASS | pyproject.toml + uv.lock present |
| canonical docs | PASS | all canonical docs present |
| tokenized caches | PASS | 21 caches, 323,057,884 tokens |
| raw Latent World JSONL | PASS | all Latent World JSONL present |
| model imports | PASS | 25 modules import OK |
| smoke: forward chunk | PASS | forward chunk shape=(256, 50257) |
| smoke: backward pass | PASS | 201 param tensors received gradients |
| smoke: checkpoint roundtrip | PASS | save+load roundtrip OK |
| phase scheduler | PASS | trainable counts {'phase0_sanity': 101750922, 'phase1_pfc_consolidation': 114688, 'phase2_early_core': 335872, 'phase3_mixed_lm': 335872} |
| eval harnesses import | PASS | all 7 prediction harnesses importable |
| disk space | PASS | 23.5 GB free |
| CUDA | PASS | skipped (--cuda not set) |
| git state | FAIL | uncommitted changes: 20 files |