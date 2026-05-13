"""Phase-aware training runner.

Usage:
  uv run python -m model.run_phase --phase 0 --steps 1000
  uv run python -m model.run_phase --phase 1 --resume checkpoints/phase0.pt --steps 1000
  uv run python -m model.run_phase --phase 2 --resume checkpoints/phase1.pt --steps 1000
  uv run python -m model.run_phase --phase 3 --resume checkpoints/phase2.pt --steps 1000

Each phase:
  - Loads/initializes model + memory state
  - Applies phase config (LoRA toggles, loss weights, data caches)
  - Builds dataset iterator from configured caches
  - Runs train_loop for --steps
  - Saves checkpoint at end
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model.checkpoint import load_checkpoint, save_checkpoint
from model.config import IPCNConfig
from model.dataset import MixedDataset, SequentialChunkDataset, TokenizedCache
from model.ipcn import IPCN
from model.phases import Phase, apply_phase, get_phase_config, trainable_param_count
from model.train import build_optimizer, train_loop


PHASES = {0: Phase.PHASE_0_SANITY, 1: Phase.PHASE_1_PFC_CONSOLIDATION,
          2: Phase.PHASE_2_EARLY_CORE_CONSOLIDATION, 3: Phase.PHASE_3_MIXED_LM}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=int, required=True, choices=[0, 1, 2, 3])
    p.add_argument("--steps", type=int, default=None, help="override default phase max_steps")
    p.add_argument("--resume", type=str, default=None, help="checkpoint to load")
    p.add_argument("--out-ckpt", type=str, default=None, help="output checkpoint path")
    p.add_argument("--log-path", type=str, default=None)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    phase = PHASES[args.phase]

    # Load or init
    if args.resume:
        model, opt_state, cfg, start_step = load_checkpoint(args.resume, map_location=args.device)
        model = model.to(args.device)
        print(f"Resumed from {args.resume} (train_step={start_step})")
    else:
        cfg = IPCNConfig()
        model = IPCN(cfg).to(args.device)
        opt_state = None
        start_step = 0

    # Apply phase config
    pc = get_phase_config(phase, cfg)
    apply_phase(model, pc)
    trainable = trainable_param_count(model)
    total = sum(p_.numel() for p_ in model.parameters())
    print(f"Phase: {pc.name}")
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print(f"Data caches: {len(pc.data_caches)}")
    for c in pc.data_caches:
        print(f"  - {c}")

    # Build optimizer
    opt = build_optimizer(model, cfg)
    if opt_state is not None:
        try:
            opt.load_state_dict(opt_state)
            print("Restored optimizer state.")
        except Exception as e:
            print(f"Could not restore optimizer state: {e}. Starting fresh.")

    # Build dataset
    caches = [TokenizedCache(prefix) for prefix in pc.data_caches if Path(prefix + ".tokens.bin").exists()]
    if not caches:
        raise RuntimeError("no caches available for this phase")
    datasets = [SequentialChunkDataset(c, chunk_length=cfg.chunk_length, shuffle_examples=True, seed=args.seed + i)
                for i, c in enumerate(caches)]
    if len(datasets) == 1:
        iterator = iter(datasets[0])
    else:
        iterator = iter(MixedDataset(datasets, seed=args.seed))

    # Train
    max_steps = args.steps if args.steps is not None else pc.max_steps
    log_path = args.log_path or f"logs/{pc.name}.jsonl"
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Training {max_steps} steps. Logging to {log_path}.")
    train_loop(model, iterator, cfg, max_steps=max_steps, log_every=args.log_every, log_path=log_path)

    # Checkpoint
    out_ckpt = args.out_ckpt or f"checkpoints/{pc.name}.pt"
    save_checkpoint(out_ckpt, model, opt, cfg, train_step=int(model.train_step.item()),
                    extra={"phase": pc.name})
    print(f"Saved checkpoint to {out_ckpt}")


if __name__ == "__main__":
    main()
