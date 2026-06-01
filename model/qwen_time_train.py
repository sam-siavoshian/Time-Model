"""Trainer for QwenTime. Passes per-conversation tau_t into the model.

Each conversation has its own tau_t (per gen_*_conversation in
qwen_time_data.py). The forward pass uses that tau to set the chrono
encoder vector, which is injected at every layer.

LM loss is masked to the assistant's response tokens only (answer-mask
pattern from Track B). The whole response is one chunk per conversation;
no cross-chunk memory in this trainer (Track C focuses on time, not
memory routing).

Usage:
  uv run python -m model.qwen_time_train --data runs/demo/data/train_v1.jsonl \
      --steps 8000 --device cuda --out runs/demo/checkpoints/qwen_time_v1.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model.qwen_time import (
    V15_BASE_MODEL_NAME,
    V15_INJECTION_TYPE,
    V15_LORA_RANK,
    QwenTime,
    QwenTimeConfig,
    build_qwen_time,
    qwen_time_checkpoint_metadata,
    qwen_time_config_dict,
)


def load_initial_trainable_state(model: QwenTime, ckpt_path: str) -> int:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    trainable_state = state.get("trainable_state")
    if not isinstance(trainable_state, dict):
        raise ValueError(f"initial checkpoint {ckpt_path} is missing trainable_state")
    current = dict(model.named_parameters())
    unexpected = sorted(name for name in trainable_state if name not in current)
    shape_mismatches = [
        f"{name}: checkpoint={tuple(tensor.shape)} current={tuple(current[name].shape)}"
        for name, tensor in trainable_state.items()
        if name in current and tuple(tensor.shape) != tuple(current[name].shape)
    ]
    if unexpected or shape_mismatches:
        parts = []
        if unexpected:
            parts.append(f"unexpected tensors={unexpected[:8]}")
        if shape_mismatches:
            parts.append(f"shape mismatches={shape_mismatches[:8]}")
        raise ValueError(f"initial checkpoint mismatch for {ckpt_path}: " + "; ".join(parts))
    for name, tensor in trainable_state.items():
        current[name].data.copy_(tensor.to(current[name].device, dtype=current[name].dtype))
    return len(trainable_state)


def checkpoint_train_step(ckpt_path: str) -> int:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return int(state.get("train_step", 0))


def trainable_state_dict(model: QwenTime) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def save_trainable_checkpoint(
    model: QwenTime,
    cfg: QwenTimeConfig,
    args: argparse.Namespace,
    step: int,
    out_path: str | Path,
) -> None:
    trainable_state = trainable_state_dict(model)
    config_metadata = qwen_time_checkpoint_metadata(cfg, tuple(model._inject_layers))
    config_metadata["trainable_names"] = sorted(trainable_state)
    save = {
        "trainable_state": trainable_state,
        "cfg": qwen_time_config_dict(cfg),
        "config_metadata": config_metadata,
        "train_args": vars(args),
        "train_step": int(step),
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(save, str(tmp))
    os.replace(str(tmp), path)


def periodic_checkpoint_path(out_path: str | Path, step: int) -> Path:
    path = Path(out_path)
    return path.with_name(f"{path.stem}.step{step}{path.suffix}")


def _proc_status_kib(field: str) -> int | None:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith(f"{field}:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def _mem_available_kib() -> int | None:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def memory_telemetry(device: str) -> dict[str, int | None]:
    telemetry: dict[str, int | None] = {
        "mem_available_kib": _mem_available_kib(),
        "proc_rss_kib": _proc_status_kib("VmRSS"),
        "proc_hwm_kib": _proc_status_kib("VmHWM"),
    }
    if str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            telemetry["cuda_allocated_bytes"] = int(torch.cuda.memory_allocated())
            telemetry["cuda_reserved_bytes"] = int(torch.cuda.memory_reserved())
            telemetry["cuda_max_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            telemetry["cuda_max_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        except Exception:
            telemetry["cuda_allocated_bytes"] = None
            telemetry["cuda_reserved_bytes"] = None
            telemetry["cuda_max_allocated_bytes"] = None
            telemetry["cuda_max_reserved_bytes"] = None
    return telemetry


def stream_records(path: str, shuffle: bool = True, seed: int = 0, repeat: bool = True):
    with open(path) as f:
        lines = f.readlines()
    if not lines:
        raise ValueError(f"empty training data: {path}")
    epoch = 0
    while True:
        epoch_lines = list(lines)
        if shuffle:
            rng = random.Random(seed + epoch)
            rng.shuffle(epoch_lines)
        for line in epoch_lines:
            yield json.loads(line)
        if not repeat:
            break
        epoch += 1


def make_chunk(tokenizer, rec, max_len: int):
    prefix_ids = tokenizer.encode(rec["prefix_text"], return_tensors="pt").squeeze(0)
    answer_ids = tokenizer.encode(rec["answer_text"], return_tensors="pt", add_special_tokens=False).squeeze(0)
    ids = torch.cat([prefix_ids, answer_ids])
    mask = torch.zeros(ids.shape[0], dtype=torch.bool)
    mask[prefix_ids.shape[0]:] = True
    if ids.shape[0] > max_len:
        # Keep the tail so the answer-mask remains valid.
        ids = ids[-max_len:]
        mask = mask[-max_len:]
    return ids, mask


def make_forced_choice_chunk(tokenizer, rec, max_len: int):
    ids = tokenizer.encode(rec["prefix_text"], return_tensors="pt").squeeze(0)
    if ids.shape[0] > max_len:
        ids = ids[-max_len:]
    return ids


def forced_choice_token_ids(tokenizer) -> list[int]:
    token_ids: list[int] = []
    for letter in ("A", "B", "C", "D"):
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if not ids:
            raise ValueError(f"could not tokenize forced-choice letter {letter!r}")
        token_ids.append(int(ids[0]))
    return token_ids


def train_step(model, opt, ids, mask, tau_t, device):
    ids = ids.to(device)
    mask = mask.to(device)
    out = model(ids, tau_t=float(tau_t))
    logits = out["logits"]
    targets = ids[1:].clone()
    targets[~mask[1:]] = -100
    if (targets >= 0).sum().item() == 0:
        return {"loss": float("nan"), "grad_norm": 0.0, "ppl": float("nan"), "n_target": 0}
    loss = F.cross_entropy(logits[:-1], targets, ignore_index=-100)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1.0
    )
    opt.step()
    return {
        "loss": float(loss.item()),
        "grad_norm": float(grad_norm.item()),
        "ppl": math.exp(min(float(loss.item()), 20)),
        "n_target": int((targets >= 0).sum().item()),
    }


def train_step_forced_choice(model, opt, ids, answer, tau_t, device, choice_token_ids):
    ids = ids.to(device)
    target_idx = {"A": 0, "B": 1, "C": 2, "D": 3}.get(str(answer))
    if target_idx is None:
        return {"loss": float("nan"), "grad_norm": 0.0, "ppl": float("nan"), "n_target": 0}
    out = model(ids, tau_t=float(tau_t))
    logits = out["logits"]
    choice_ids = torch.tensor(choice_token_ids, device=logits.device, dtype=torch.long)
    scores = logits[-1].index_select(0, choice_ids).unsqueeze(0)
    target = torch.tensor([target_idx], device=logits.device, dtype=torch.long)
    loss = F.cross_entropy(scores.float(), target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1.0
    )
    opt.step()
    return {
        "loss": float(loss.item()),
        "grad_norm": float(grad_norm.item()),
        "ppl": math.exp(min(float(loss.item()), 20)),
        "n_target": 1,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log-path", type=str, default=None)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--init-checkpoint", type=str, default=None,
                   help="Optional trainable-state checkpoint to load before training. "
                        "Used for explicit fine-tuning experiments; output checkpoint "
                        "is still written to --out.")
    p.add_argument("--resume-checkpoint", type=str, default=None,
                   help="Resume model weights from a trainable-state checkpoint and "
                        "continue step numbering from its train_step metadata. Optimizer "
                        "state is intentionally not restored.")
    p.add_argument("--save-every", type=int, default=0,
                   help="If >0, write periodic checkpoints every N completed steps as "
                        "<out>.stepN before the final checkpoint.")
    p.add_argument("--empty-cache-every", type=int, default=0,
                   help="If >0 on CUDA, call torch.cuda.empty_cache() every N "
                        "completed steps. Useful on unified-memory systems where "
                        "reserved CUDA cache can starve system RAM.")
    p.add_argument("--run-id", type=str, default=None,
                   help="Write checkpoint/logs under runs/<run-id>/ when --out is omitted.")
    p.add_argument("--chunk-length", type=int, default=512)
    p.add_argument("--loss-mode", choices=("assistant_ce", "forced_choice"), default="assistant_ce",
                   help="assistant_ce trains on masked assistant tokens. forced_choice "
                        "uses a four-way softmax over next-token A/B/C/D letters.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base", type=str, default=V15_BASE_MODEL_NAME)
    p.add_argument("--unfreeze-base", action="store_true")
    p.add_argument("--timescales", type=str, default="",
                   help="Comma-separated chrono timescales in seconds. Empty = use canonical v15 defaults.")
    p.add_argument("--freeze-alpha", action="store_true",
                   help="Lock per-layer chrono alpha gates at 0 throughout training. "
                        "This is the LoRA-only ablation: chrono encoder + projectors "
                        "still exist and receive gradients but cannot influence the "
                        "residual stream. Used to falsify the architectural claim by "
                        "comparing v15 (alpha learns) vs alpha=0-frozen.")
    p.add_argument("--inject-layers", type=str, default="",
                   help="Comma-separated layer indices for chrono injection. "
                        "Empty = inject at every decoder layer except the final layer (default v15). "
                        "'0' = inject only at layer 0 (L0-only ablation).")
    p.add_argument("--injection-type", type=str, default=V15_INJECTION_TYPE,
                   choices=["film", "additive"],
                   help="'film' (default v15, DiT AdaLN-Zero) or 'additive' "
                        "(pure residual chi-projected, no h-dependent scaling -- "
                        "GazeQwen-style ablation).")
    p.add_argument("--additive-beta-init", type=float, default=0.0,
                   help="When --injection-type=additive, init to_beta.bias to "
                        "this constant so beta(chi) at step 0 is non-zero and "
                        "d_out/d_alpha = beta is non-zero. Allows the additive "
                        "variant to escape the AdaLN-Zero gradient trap. "
                        "Default 0.0 = original AdaLN-Zero (which traps). "
                        "Used to test reviewer concern W9.")
    p.add_argument("--freeze-lora", action="store_true",
                   help="Lock LoRA A and B matrices at their init throughout "
                        "training. This is the chrono-only ablation: the chrono "
                        "encoder + per-layer FiLM gates still train, but the LoRA "
                        "surface capacity is frozen. Used to test whether the "
                        "chrono channel alone (without LoRA) is sufficient.")
    p.add_argument("--lora-rank", type=int, default=V15_LORA_RANK,
                   help="LoRA adapter rank. Default 8 (CI default). Used by W6 fix: "
                        "rank-bumped prompt baseline at matched total trainable params.")
    p.add_argument("--use-ia3", action="store_true",
                   help="W9 reviewer fix: use IA3 (Liu 2022) multiplicative-scaling "
                        "PEFT instead of LoRA. Wraps k_proj, v_proj, and FFN up_proj "
                        "with per-output-feature learnable scales initialized to one.")
    args = p.parse_args()
    if args.init_checkpoint and args.resume_checkpoint:
        raise SystemExit("pass only one of --init-checkpoint or --resume-checkpoint")

    if args.out is None:
        if args.run_id is None:
            raise SystemExit("output scope required: pass --out or --run-id")
        args.out = str(Path("runs") / args.run_id / "checkpoints" / "qwen_time.pt")
    if args.log_path is None:
        if args.run_id is not None:
            args.log_path = str(Path("runs") / args.run_id / "logs" / "qwen_time.jsonl")
        else:
            args.log_path = str(Path(args.out).with_suffix(".jsonl"))

    torch.manual_seed(args.seed)
    Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    cfg = QwenTimeConfig()
    cfg.base_model_name = args.base
    cfg.chunk_length = args.chunk_length
    cfg.unfreeze_base = args.unfreeze_base
    if args.lora_rank != V15_LORA_RANK:
        cfg.lora_rank = args.lora_rank
        print(f"  Override lora_rank: {cfg.lora_rank}")
    if args.use_ia3:
        cfg.use_ia3 = True
        print(f"  Override: using IA3 instead of LoRA (W9 baseline)")
    if args.timescales:
        cfg.timescales = tuple(int(x) for x in args.timescales.split(","))
        print(f"  Override timescales: {cfg.timescales}")
    if args.inject_layers:
        cfg.inject_layers = tuple(int(x) for x in args.inject_layers.split(","))
        print(f"  Override inject_layers: {cfg.inject_layers}")
    if args.injection_type and args.injection_type != "film":
        cfg.injection_type = args.injection_type
        print(f"  Override injection_type: {cfg.injection_type}")
    if args.additive_beta_init != 0.0:
        cfg.additive_beta_init = args.additive_beta_init
        print(f"  Override additive_beta_init: {cfg.additive_beta_init}")
    print(f"Loading QwenTime ({cfg.base_model_name})...")
    t0 = time.time()
    model = build_qwen_time(cfg)
    model = model.to(args.device)
    print(f"  loaded in {time.time() - t0:.1f}s")
    n_train = model.trainable_parameter_count()
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  trainable: {n_train:,} / {n_total:,} ({100*n_train/n_total:.3f}%)")
    print(f"  chrono injectors: {len(model.chrono_injectors)}")
    print(f"  LoRA modules: {model._n_lora_modules}")

    start_step = 0
    if args.resume_checkpoint:
        n_loaded = load_initial_trainable_state(model, args.resume_checkpoint)
        start_step = checkpoint_train_step(args.resume_checkpoint)
        print(
            f"  resumed trainable state from {args.resume_checkpoint} "
            f"({n_loaded} tensors, start_step={start_step})"
        )
    elif args.init_checkpoint:
        n_loaded = load_initial_trainable_state(model, args.init_checkpoint)
        print(f"  loaded initial trainable state from {args.init_checkpoint} ({n_loaded} tensors)")

    if args.freeze_alpha:
        # LoRA-only ablation: lock all per-layer alpha gates at 0 and
        # remove them from the optimizer. Chrono encoder + projectors
        # still exist but cannot affect the residual stream.
        for inj in model.chrono_injectors.values():
            with torch.no_grad():
                inj.alpha.zero_()
            inj.alpha.requires_grad_(False)
        print(f"  FROZE all {len(model.chrono_injectors)} chrono alpha gates at 0 (LoRA-only ablation)")

    if args.freeze_lora:
        # chrono-only ablation: chrono encoder + per-layer FiLM gates train,
        # but LoRA A and B matrices are locked at their init. Tests whether
        # the chrono channel alone (without LoRA surface capacity) suffices.
        n_lora_frozen = 0
        for n, p in model.named_parameters():
            if "lora_A" in n or "lora_B" in n:
                p.requires_grad_(False)
                n_lora_frozen += 1
        print(f"  FROZE all {n_lora_frozen} LoRA A/B parameters (chrono-only ablation)")

    if args.unfreeze_base:
        # Split LR: LoRA + injectors at args.lr, base params at 1/100th
        # to prevent catastrophic forgetting / weight collapse.
        lora_or_inj = []
        base_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "lora_" in n or "chrono" in n or "alpha" in n or "to_gamma" in n or "to_beta" in n:
                lora_or_inj.append(p)
            else:
                base_params.append(p)
        print(f"  unfrozen base: split LR groups -- lora/inj={len(lora_or_inj)} base={len(base_params)}")
        opt = torch.optim.AdamW(
            [{"params": lora_or_inj, "lr": args.lr},
             {"params": base_params, "lr": args.lr * 0.01}],
            weight_decay=0.01,
        )
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    log_f = open(args.log_path, "a" if args.resume_checkpoint else "w", buffering=1)
    print(f"Training from step {start_step} to {args.steps}. Log -> {args.log_path}")
    step = start_step
    iterator = stream_records(args.data, seed=args.seed)
    choice_token_ids = forced_choice_token_ids(model.tokenizer) if args.loss_mode == "forced_choice" else None
    for _ in range(start_step):
        next(iterator)
    t_train = time.time()
    for rec in iterator:
        if step >= args.steps:
            break
        try:
            if args.loss_mode == "forced_choice":
                ids = make_forced_choice_chunk(model.tokenizer, rec, args.chunk_length)
                mask = None
            else:
                ids, mask = make_chunk(model.tokenizer, rec, args.chunk_length)
        except Exception as e:
            continue
        if args.loss_mode == "forced_choice":
            out = train_step_forced_choice(
                model,
                opt,
                ids,
                rec.get("answer"),
                rec.get("tau_t", 0.0),
                args.device,
                choice_token_ids,
            )
        else:
            out = train_step(model, opt, ids, mask, rec.get("tau_t", 0.0), args.device)
        out.update({
            "step": step,
            "mode": rec.get("mode", "?"),
            "loss_mode": args.loss_mode,
            "tau_t": rec.get("tau_t", 0.0),
            "time": time.time(),
            "memory": memory_telemetry(args.device),
        })
        log_f.write(json.dumps(out) + "\n")
        if step % args.log_every == 0:
            mem = out["memory"]
            mem_gib = (
                f"{mem['mem_available_kib'] / (1024 ** 2):.1f}GiB"
                if mem.get("mem_available_kib") is not None
                else "n/a"
            )
            print(
                f"step={step:6d} | loss={out['loss']:7.4f} ppl={out.get('ppl', 0):8.1f} "
                f"grad={out['grad_norm']:6.3f} mode={out['mode']:10s} tau={out['tau_t']:.1f} "
                f"tgt={out.get('n_target', 0):3d} mem_avail={mem_gib}"
            )
        step += 1
        if (
            args.empty_cache_every > 0
            and step % args.empty_cache_every == 0
            and str(args.device).startswith("cuda")
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()
        if args.save_every > 0 and step % args.save_every == 0:
            ckpt_path = periodic_checkpoint_path(args.out, step)
            save_trainable_checkpoint(model, cfg, args, step, ckpt_path)
            print(f"Saved periodic checkpoint to {ckpt_path}")
    log_f.write(json.dumps({"event": "training_complete", "step": step,
                            "max_steps": args.steps, "reason": "max_steps",
                            "time": time.time()}) + "\n")
    log_f.close()

    save_trainable_checkpoint(model, cfg, args, step, args.out)
    print(f"Saved checkpoint to {args.out}")
    print(f"Total wall time: {time.time() - t_train:.1f}s")


if __name__ == "__main__":
    main()
