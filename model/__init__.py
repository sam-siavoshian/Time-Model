"""Chronometric Injection model package.

Current modules (post-v15 reframe to Track C: QwenTime FiLM injection):
  qwen_time              Frozen Qwen2.5-3B + LoRA + per-layer FiLM chrono injection
  qwen_time_check        T1-T4 evaluation harness for internal validation
  qwen_time_train        Training loop for v15 anchor + cross-seed runs
  qwen_time_data         Synthetic conversation generator (latent world)
  qwen_time_probe        Linear probe for tau-axis representation
  qwen_time_falsify      Causal alpha-flip falsification battery
  qwen_time_pressure_v2  Deadline-pressure adaptive-length tests
  qwen_time_extra_controls  Paraphrase + half-flip + teacher-forced T4 controls

Legacy IPCN (Track A 102M from-scratch / Track B memory routing) is archived
under `docs/history/` and `reports/archive/`. This package no longer exposes
any top-level symbols; import submodules directly.
"""
