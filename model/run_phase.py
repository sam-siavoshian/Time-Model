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
from model.latent_world_loader import LatentWorldChunkDataset
from model.phases import Phase, apply_phase, get_phase_config, trainable_param_count
from model.replay_buffer import ReplayBuffer
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
    p.add_argument("--use-real-tau", action="store_true",
                   help="use LatentWorldChunkDataset (real tau from event metadata, slower)")
    p.add_argument("--enable-consolidation", action="store_true",
                   help="enable consolidation passes during training (Phase 1+)")
    p.add_argument("--cons-freq", type=int, default=None,
                   help="override consolidation_frequency (chunks between passes)")
    p.add_argument("--t-stable-override", type=int, default=None,
                   help="override cfg.t_stable for smoke testing (default 512)")
    p.add_argument("--tau-cons-override", type=float, default=None,
                   help="override cfg.tau_cons eligibility threshold for smoke testing (default 3.0)")
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

    # Override thresholds for smoke testing if requested
    if args.cons_freq is not None:
        cfg.consolidation_frequency = args.cons_freq
    if args.t_stable_override is not None:
        cfg.t_stable = args.t_stable_override
    if args.tau_cons_override is not None:
        cfg.tau_cons = args.tau_cons_override

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
    datasets = []
    for i, prefix in enumerate(pc.data_caches):
        if args.use_real_tau and "/latent_world/" in prefix:
            # Map tokenized cache prefix back to JSONL: e.g.
            # data/tokenized/latent_world/train_1k -> data/latent_world/train_1k.jsonl
            split = prefix.rsplit("/", 1)[-1]
            jsonl_path = f"data/latent_world/{split}.jsonl"
            if Path(jsonl_path).exists():
                datasets.append(LatentWorldChunkDataset(
                    jsonl_path, chunk_length=cfg.chunk_length, shuffle=True, seed=args.seed + i
                ))
                continue
        if Path(prefix + ".tokens.bin").exists():
            cache = TokenizedCache(prefix)
            datasets.append(SequentialChunkDataset(
                cache, chunk_length=cfg.chunk_length, shuffle_examples=True, seed=args.seed + i
            ))
    if not datasets:
        raise RuntimeError("no datasets available for this phase")
    if len(datasets) == 1:
        iterator = iter(datasets[0])
    else:
        iterator = iter(MixedDataset(datasets, seed=args.seed))

    # Train
    max_steps = args.steps if args.steps is not None else pc.max_steps
    log_path = args.log_path or f"logs/{pc.name}.jsonl"
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Training {max_steps} steps. Logging to {log_path}.")
    print(f"Real-tau extraction: {args.use_real_tau}")
    print(f"Consolidation: {args.enable_consolidation}")

    replay_buffer = None
    if args.enable_consolidation:
        replay_buffer = ReplayBuffer(n_slots=cfg.n_slots, capacity_per_slot=256, seed=args.seed)

    train_loop(
        model, iterator, cfg,
        max_steps=max_steps,
        log_every=args.log_every,
        log_path=log_path,
        enable_consolidation=args.enable_consolidation,
        replay_buffer=replay_buffer,
    )

    # Checkpoint
    out_ckpt = args.out_ckpt or f"checkpoints/{pc.name}.pt"
    save_checkpoint(out_ckpt, model, opt, cfg, train_step=int(model.train_step.item()),
                    extra={"phase": pc.name})
    print(f"Saved checkpoint to {out_ckpt}")


if __name__ == "__main__":
    main()
