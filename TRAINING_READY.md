# IPCN — Training-Ready (Final)

Date: 2026-05-13
Latest commit: `24c7308`
Preflight: **13/13 PASS** (see `reports/preflight.md`)

---

## Production tooling shipped

| Component | Path | Purpose |
|---|---|---|
| Preflight gate | `scripts/preflight.py` | 13-check readiness audit. Block training if anything fails. |
| Monitor | `scripts/monitor.py` | Tail training log, rolling metrics, alert on spikes/NaN/stalls |
| Safety watchdog | `scripts/safety.py` | Kill training PID on LM explosion, NaN, disk-full |
| Tmux launcher | `scripts/tmux_launch.sh` | 3-pane session: training + safety + monitor |
| Cleanup | `scripts/cleanup.sh` | Remove __pycache__, stale logs, scratch checkpoints |
| Feature importance | `scripts/feature_importance.py` | Per-loss %, per-group grads, slot utilization |
| E2E smoke | `scripts/e2e_smoke.sh` | Phase 0 → Phase 1 → eval cycle on CPU |
| Path conventions | `PATHS.md` | Canonical filesystem layout + sync rules |

---

## Train-readiness gate

```bash
uv run python -m scripts.preflight
# On Spark also: uv run python -m scripts.preflight --cuda
```

Current state on laptop (CPU): **13/13 PASS**.

---

## Launch sequence on DGX Spark

### Step 1: Sync (laptop → Spark)
```bash
cd "/Users/samsiavoshian/Desktop/Coding Stuff/Time-Model"
bash scripts/cleanup.sh
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='logs' \
  --exclude='checkpoints' --exclude='reports/e2e_smoke' \
  ./ root1@100.83.86.5:Desktop/Time-Model/
rsync -avz data/ root1@100.83.86.5:Desktop/Time-Model/data/
```

### Step 2: Install deps + preflight on Spark
```bash
ssh root1@100.83.86.5 << 'EOF'
cd Desktop/Time-Model
export PATH="$HOME/.local/bin:$PATH"
uv sync
uv run python -m scripts.preflight --cuda
EOF
```

Expected: 13/13 PASS including CUDA check.

### Step 3: Launch Phase 0 in tmux
```bash
ssh root1@100.83.86.5
cd Desktop/Time-Model
export PATH="$HOME/.local/bin:$PATH"
bash scripts/tmux_launch.sh phase0 100000 cuda
tmux attach -t ipcn
```

Tmux session `ipcn` opens with:
- **Pane 0:** training (`run_phase --phase 0 --steps 100000 --device cuda --use-real-tau --ckpt-every 10000`)
- **Pane 1:** safety (kills training on NaN / explosion / disk-full)
- **Pane 2:** live monitor (rolling LM, gradient norm, alerts)

Detach with `Ctrl+b d`. Reattach with `tmux attach -t ipcn`.

### Step 4: Watch + intervene
- Logs stream to `logs/phase0_sanity.jsonl`
- Intermediate checkpoints every 10k steps: `checkpoints/phase0_sanity_step{N}.pt`
- Alerts in `reports/alerts.jsonl`
- If LM regresses or safety kills: investigate, adjust, restart via `--resume`

### Step 5: Eval Phase 0 result
```bash
uv run python -m model.eval_all \
  --checkpoint checkpoints/phase0_sanity.pt \
  --n-trials 100 \
  --out reports/phase0.md \
  --device cuda
```

Report: `reports/phase0.md`. Predictions H1/H2/H5/H6/H7 should trend positive vs baseline.

### Step 6: Feature importance + ablation
```bash
uv run python -m scripts.feature_importance --checkpoint checkpoints/phase0_sanity.pt
uv run python -m model.ablation_runner --steps 5000 --variants A0 A1 A2 A3 A4 A5 A6
```

### Step 7: Chain Phase 1 → 2 → 3
```bash
bash scripts/tmux_launch.sh phase1 50000 cuda checkpoints/phase0_sanity.pt
# After Phase 1: tmux attach -t ipcn -> Ctrl+b d
bash scripts/tmux_launch.sh phase2 50000 cuda checkpoints/phase1_pfc_consolidation.pt
bash scripts/tmux_launch.sh phase3 100000 cuda checkpoints/phase2_early_core.pt
```

### Step 8: Final eval + H4 CTI
```bash
uv run python -m model.eval_all \
  --checkpoint checkpoints/phase3_mixed_lm.pt \
  --pre-checkpoint checkpoints/phase0_sanity.pt \
  --n-trials 200 \
  --out reports/final.md \
  --device cuda
```

### Step 9: Sync results back to laptop
```bash
rsync -avz root1@100.83.86.5:Desktop/Time-Model/checkpoints/ ./checkpoints/
rsync -avz root1@100.83.86.5:Desktop/Time-Model/logs/ ./logs/
rsync -avz root1@100.83.86.5:Desktop/Time-Model/reports/ ./reports/
```

---

## Safety guarantees

- **Stop-loss watchdog** kills training on LM > 10× trailing mean, NaN/Inf, disk-full
- **Validation gates** rollback consolidation pass if LM drift > 0.02 or accuracy drops > 2%
- **Snapshot/restore** of LoRA adapters on failed consolidation
- **Intermediate checkpoints** every 10k steps for crash recovery
- **Resume** via `run_phase --resume <ckpt>` from any saved state
- **Monitor alerts** to `reports/alerts.jsonl` for external paging

---

## Model file destinations

Per `PATHS.md`:

| File | Path |
|---|---|
| Phase 0 final | `checkpoints/phase0_sanity.pt` |
| Phase 1 final | `checkpoints/phase1_pfc_consolidation.pt` |
| Phase 2 final | `checkpoints/phase2_early_core.pt` |
| Phase 3 final | `checkpoints/phase3_mixed_lm.pt` |
| Intermediates | `checkpoints/<phase>_step<N>.pt` |
| Logs | `logs/<phase_name>.jsonl` |
| Reports | `reports/<phase_name>.{md,json}` |

Final phase checkpoint size: ~410 MB.
Intermediate: ~1.2 GB (with optimizer state).
Total Phase 0-3 disk: ~45 GB. Spark has 228 GB; comfortable.

---

## Training monitoring

Three layers:

1. **Inline log** (`logs/<phase>.jsonl`): structured JSONL, one record per training step. Includes LM, total, all 9 loss components, grad norm, memory norm, z norm, chunk time, example/chunk indices, and consolidation events.

2. **Live monitor** (`scripts/monitor.py --tail`): rolling LM avg, perplexity, alerts on spikes. Runs in tmux pane 2.

3. **Safety watchdog** (`scripts/safety.py`): kills training if hard failure. Runs in tmux pane 1.

After training, `scripts/feature_importance.py` produces post-hoc analysis: loss contribution %, gradient magnitudes per parameter group, slot utilization statistics.

---

## Resume system

Any training run can be resumed from any saved checkpoint:

```bash
uv run python -m model.run_phase \
  --phase <N> --steps <remaining> \
  --resume <ckpt> \
  --device cuda \
  --out-ckpt <new_ckpt>
```

Intermediate checkpoints (`<phase>_step<N>.pt`) include optimizer state. Phase final checkpoints (`<phase_name>.pt`) include everything needed for next-phase resume.

If training is killed by safety watchdog or crashes, the latest intermediate checkpoint is the resume point. The runner rebuilds the optimizer fresh when param groups change across phases (warning is benign).

---

## What's verified before Spark

- **Preflight 13/13 PASS** (smoke forward+backward+checkpoint, phase scheduler, all eval imports, disk space, git state)
- **E2E smoke** (`scripts/e2e_smoke.sh`) runs full Phase 0 → Phase 1 → eval on CPU
- **Feature importance baseline** (`reports/feature_importance.md`) on untrained model
- **Untrained eval baseline** (`reports/untrained_baseline_v2.md`) for diff after training
- **Ablation matrix** (`scripts/feature_importance.py` + `model/ablation_runner.py`) works on 7 variants

---

## Ready to train? YES

13/13 preflight checks green. All datasets, models, training, evals, safety, monitoring in place. Phase 0 launch is one command on Spark.
