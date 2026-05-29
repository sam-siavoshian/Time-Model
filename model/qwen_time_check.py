"""Time-awareness eval for QwenTime.

Three falsifiable tests aligned with the paper's four time properties.

T1 CLOCK CONSISTENCY (Gap 1: no clock):
  For 50 trials, query the model with the SAME prompt 'how long has it
  been?' at 8 different tau values {15s, 1m, 5m, 30m, 2h, 12h, 1d, 1w}.
  Parse the response into seconds. Compute Pearson r between predicted
  and true tau.
  Pass threshold: r >= 0.8. Falsify: r <= 0.3.

T2 SILENT-GAP ACK (Gap 2: no silent-gap awareness):
  50 pairs of two-turn conversations. Pair A has delta_tau = 10s; pair
  B has delta_tau = 86400s. Generate assistant's second response. Count
  how often response contains an ack ('welcome back', 'been a while',
  duration mention).
  Pass threshold: ack_rate(B) - ack_rate(A) >= 0.5.

T3 PHASE DISCRIMINATION (property 3: multi-scale phase):
  50 trials per condition. Same prompt 'good morning' presented at
  tau corresponding to weekday vs weekend. Measure KL divergence
  between the response distributions or word-overlap similarity.
  Pass threshold: KL >= 0.3 OR weekend_keyword_rate > weekday_keyword_rate by 0.3.

T4 (bonus) MUTABILITY: confirm chrono encoder is causally hooked up.
  For one fixed prompt, sweep tau across the 8 values and verify
  output logits differ. Mean pairwise KL across tau values > 0.05.
  This is the negative control: if T4 passes but T1-T3 fail, chrono
  signal reaches output but model hasn't learned to USE it.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import random
import re
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
    qwen_time_config_dict,
)


# When True, every prompt fed to the model gets a `[elapsed: X]` prefix
# injected after the first `<|im_start|>user\n` marker. This mirrors the
# training distribution for prompt_baseline ckpts (trained via
# qwen_time_data_prompt.inject_tau_in_text). Set from CLI flag --inject-prompt.
INJECT_PROMPT_TAU = False
REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_METADATA_REGISTRY = REPO_ROOT / "checkpoints" / "metadata_registry.json"


PROMPT_FORMAT = "elapsed"  # set from --prompt-format CLI


def _tau_text(tau: float) -> str:
    """Format tau according to global PROMPT_FORMAT setting.

    'elapsed' (default): bracketed relative duration '[elapsed: 3h 42m]'
    'iso': ISO timestamp '[timestamp: 2024-01-15T14:42:00Z]'
    'nl': natural language 'About 3 hours and 42 minutes have passed
          since we started.'

    Format must match the format used in training data.
    """
    if PROMPT_FORMAT == "iso":
        import datetime
        ref = datetime.datetime(2024, 1, 15, 14, 42, 0, tzinfo=datetime.timezone.utc)
        t = ref + datetime.timedelta(seconds=float(tau))
        return f"[timestamp: {t.strftime('%Y-%m-%dT%H:%M:%SZ')}]"
    if PROMPT_FORMAT == "nl":
        if tau < 60:
            return f"About {tau:.0f} seconds have passed since we started."
        if tau < 3600:
            return f"About {tau/60:.0f} minutes have passed since we started."
        if tau < 86400:
            h = int(tau // 3600); m = int((tau % 3600) // 60)
            return f"About {h} hours and {m} minutes have passed since we started."
        d = int(tau // 86400); h = int((tau % 86400) // 3600)
        return f"About {d} days and {h} hours have passed since we started."
    # default 'elapsed'
    if tau < 60:
        return f"[elapsed: {tau:.1f}s]"
    if tau < 3600:
        m = tau / 60
        return f"[elapsed: {m:.1f}m]"
    if tau < 86400:
        h = int(tau // 3600); m = int((tau % 3600) // 60)
        return f"[elapsed: {h}h {m}m]"
    d = int(tau // 86400); h = int((tau % 86400) // 3600)
    return f"[elapsed: {d}d {h}h]"


def _maybe_inject_prompt(prompt: str, tau: float) -> str:
    """If INJECT_PROMPT_TAU global is set, prepend `[elapsed: X] ` after the
    first `<|im_start|>user\n` marker. Matches training prep exactly
    (qwen_time_data_prompt.inject_tau_in_text uses replace(..., 1))."""
    if not INJECT_PROMPT_TAU:
        return prompt
    marker = "<|im_start|>user\n"
    if marker not in prompt:
        return prompt
    return prompt.replace(marker, f"{marker}{_tau_text(float(tau))} ", 1)


def _normalize_cfg_value(value):
    if isinstance(value, list):
        return tuple(value)
    return value


def _deep_merge_cfg(defaults: dict, overrides: dict | None) -> dict:
    cfg = dict(defaults)
    if overrides:
        cfg.update(overrides)
    return cfg


def _checkpoint_relpath(ckpt_path: str) -> str:
    path = Path(ckpt_path)
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _load_checkpoint_registry() -> dict | None:
    if not CHECKPOINT_METADATA_REGISTRY.exists():
        return None
    with CHECKPOINT_METADATA_REGISTRY.open() as f:
        return json.load(f)


def _sidecar_metadata_for_checkpoint(ckpt_path: str) -> dict | None:
    registry = _load_checkpoint_registry()
    if not registry:
        return None
    rel = _checkpoint_relpath(ckpt_path)
    defaults = registry.get("defaults", {})
    for entry in registry.get("entries", []):
        exact = entry.get("path")
        pattern = entry.get("glob")
        if (exact and rel == exact) or (pattern and fnmatch.fnmatch(rel, pattern)):
            return {
                "cfg": _deep_merge_cfg(defaults, entry.get("cfg")),
                "status": entry.get("status"),
                "layer_policy": entry.get("layer_policy"),
                "caveat": entry.get("caveat"),
                "registry_match": exact or pattern,
            }
    return None


def _extract_checkpoint_cfg(state: dict, ckpt_path: str) -> dict | None:
    metadata = state.get("config_metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("cfg"), dict):
        return metadata["cfg"]
    cfg = state.get("cfg")
    if isinstance(cfg, dict):
        return cfg
    sidecar = _sidecar_metadata_for_checkpoint(ckpt_path)
    if sidecar and isinstance(sidecar.get("cfg"), dict):
        print(f"  using sidecar checkpoint metadata: {sidecar.get('registry_match')}")
        if sidecar.get("caveat"):
            print(f"  sidecar caveat: {sidecar['caveat']}")
        return sidecar["cfg"]
    return None


def _format_mismatch(field: str, checkpoint_value, model_value) -> str:
    return f"{field}: checkpoint={checkpoint_value!r} current={model_value!r}"


def validate_checkpoint_compatible(
    model: QwenTime,
    state: dict,
    ckpt_path: str,
    allow_unregistered_legacy_ckpt: bool = False,
):
    """Validate checkpoint architecture metadata before copying tensors."""
    ckpt_cfg = _extract_checkpoint_cfg(state, ckpt_path)
    if ckpt_cfg is None:
        if allow_unregistered_legacy_ckpt:
            print("  warning: checkpoint has no config metadata or sidecar registry entry; validating tensors only")
        else:
            raise ValueError(
                "Checkpoint has no embedded config metadata and no sidecar registry entry: "
                f"{ckpt_path}. Add it to checkpoints/metadata_registry.json or pass "
                "--allow-unregistered-legacy-ckpt for manual provenance work."
            )
    else:
        model_cfg = qwen_time_config_dict(model.cfg)
        fields = (
            "base_model_name",
            "timescales",
            "inject_layers",
            "lora_rank",
            "lora_layers",
            "lora_targets",
            "lora_lm_head",
            "unfreeze_base",
            "chunk_length",
            "injection_type",
            "additive_beta_init",
            "use_ia3",
        )
        mismatches = []
        for field in fields:
            if field not in ckpt_cfg:
                continue
            ckpt_value = _normalize_cfg_value(ckpt_cfg[field])
            model_value = _normalize_cfg_value(model_cfg[field])
            if ckpt_value != model_value:
                mismatches.append(_format_mismatch(field, ckpt_value, model_value))
        metadata = state.get("config_metadata")
        if isinstance(metadata, dict) and "resolved_inject_layers" in metadata:
            ckpt_layers = tuple(metadata["resolved_inject_layers"])
            model_layers = tuple(getattr(model, "_inject_layers", ()))
            if ckpt_layers != model_layers:
                mismatches.append(_format_mismatch("resolved_inject_layers", ckpt_layers, model_layers))
        if mismatches:
            joined = "; ".join(mismatches)
            raise ValueError(f"Checkpoint config mismatch for {ckpt_path}: {joined}")

    trainable_state = state.get("trainable_state")
    if not isinstance(trainable_state, dict):
        raise ValueError(f"Checkpoint {ckpt_path} is missing a trainable_state dict")

    current_params = dict(model.named_parameters())
    unexpected = sorted(name for name in trainable_state if name not in current_params)
    metadata = state.get("config_metadata")
    expected_trainable = None
    if isinstance(metadata, dict) and isinstance(metadata.get("trainable_names"), list):
        expected_trainable = set(metadata["trainable_names"])
    missing = sorted(expected_trainable - set(trainable_state)) if expected_trainable is not None else []
    shape_mismatches = []
    for name, tensor in trainable_state.items():
        if name not in current_params:
            continue
        if tuple(tensor.shape) != tuple(current_params[name].shape):
            shape_mismatches.append(
                f"{name}: checkpoint={tuple(tensor.shape)} current={tuple(current_params[name].shape)}"
            )
    if unexpected or missing or shape_mismatches:
        parts = []
        if unexpected:
            parts.append(f"unexpected tensors={unexpected[:8]}")
        if missing:
            parts.append(f"missing tensors={missing[:8]}")
        if shape_mismatches:
            parts.append(f"shape mismatches={shape_mismatches[:8]}")
        raise ValueError(f"Checkpoint tensor mismatch for {ckpt_path}: " + "; ".join(parts))


def load_trainable(model: QwenTime, ckpt_path: str, allow_unregistered_legacy_ckpt: bool = False):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    validate_checkpoint_compatible(model, state, ckpt_path, allow_unregistered_legacy_ckpt)
    cur = dict(model.named_parameters())
    n = 0
    for name, t in state["trainable_state"].items():
        cur[name].data.copy_(t.to(cur[name].dtype))
        n += 1
    print(f"  loaded {n} trainable tensors")


@torch.no_grad()
def greedy_decode(model: QwenTime, prompt: str, tau_t: float,
                  max_new_tokens: int = 24, device: str = "cuda",
                  temperature: float = 0.0, top_p: float = 1.0,
                  seed: int = None) -> str:
    """Greedy by default (temperature=0). If temperature > 0, use
    temperature + top-p sampling with optional torch seed. This is the
    eff-n=1 fix path: T2/T3 can be re-run with temperature=0.7 + 30
    seeds to get real variance instead of replicate-inflated n=30."""
    tok = model.tokenizer
    prompt = _maybe_inject_prompt(prompt, tau_t)
    ids = tok.encode(prompt, return_tensors="pt").squeeze(0).to(device)
    generated = []
    cur = ids
    im_end = tok.convert_tokens_to_ids("<|im_end|>") if hasattr(tok, "convert_tokens_to_ids") else None
    if seed is not None:
        torch.manual_seed(seed)
    for _ in range(max_new_tokens):
        out = model(cur, tau_t=tau_t)
        logits = out["logits"][-1].float()
        if temperature <= 0.0:
            next_id = int(logits.argmax().item())
        else:
            scaled = logits / temperature
            probs = torch.softmax(scaled, dim=-1)
            if 0.0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum > top_p
                mask[0] = False                                # keep top
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum()
                next_id = int(sorted_idx[torch.multinomial(sorted_probs, 1).item()].item())
            else:
                next_id = int(torch.multinomial(probs, 1).item())
        if im_end is not None and next_id == im_end:
            break
        if next_id == tok.eos_token_id:
            break
        generated.append(next_id)
        cur = torch.cat([cur, torch.tensor([next_id], device=device)])
    return tok.decode(generated)


@torch.no_grad()
def logits_at_first_pos(model: QwenTime, prompt: str, tau_t: float, device: str = "cuda") -> torch.Tensor:
    tok = model.tokenizer
    prompt = _maybe_inject_prompt(prompt, tau_t)
    ids = tok.encode(prompt, return_tensors="pt").squeeze(0).to(device)
    out = model(ids, tau_t=tau_t)
    return out["logits"][-1].float().cpu()                     # (vocab,)


def parse_duration_to_seconds(s: str) -> float:
    """Word-boundary regex. Bare 'h' / 'm' / 's' / 'd' single-letter
    aliases dropped because they match too aggressively (e.g. 'a' in
    'have' could trigger). Order matters: weeks first, seconds last."""
    s = s.lower()
    patterns = [
        (r"(\d+(?:\.\d+)?)\s*weeks?\b", 604800.0),
        (r"(\d+(?:\.\d+)?)\s*days?\b", 86400.0),
        (r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\b", 3600.0),
        (r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)\b", 60.0),
        (r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?)\b", 1.0),
    ]
    for pat, mult in patterns:
        m = re.search(pat, s)
        if m:
            return float(m.group(1)) * mult
    return float("nan")


def t1b_clock_ood(model, device, n=24):
    """Held-out tau drawn from disjoint values NEVER seen in training
    grid. Pearson r + log-MAE on these is the load-bearing test for
    "model learned a continuous duration map, not a bucket lookup."
    """
    import math, random, statistics
    rng = random.Random(7777)
    prompt = "<|im_start|>user\nHow long has it been since we started?<|im_end|>\n<|im_start|>assistant\n"
    pred = []; truth = []; log_errs = []
    for _ in range(n):
        # Sample log-uniform from broader range than training included
        tau = math.exp(rng.uniform(math.log(2.0), math.log(14 * 86400.0)))
        resp = greedy_decode(model, prompt, tau_t=tau, device=device)
        sec = parse_duration_to_seconds(resp)
        if sec == sec and sec > 0:
            pred.append(sec); truth.append(tau)
            log_errs.append(abs(math.log10(sec) - math.log10(tau)))
    if len(pred) < 4:
        return {"n": len(pred), "pearson_r": float("nan"), "log_mae": float("nan")}
    mp, mt = statistics.mean(pred), statistics.mean(truth)
    num = sum((p - mp) * (t - mt) for p, t in zip(pred, truth))
    denom = (sum((p - mp) ** 2 for p in pred) * sum((t - mt) ** 2 for t in truth)) ** 0.5
    r = num / denom if denom > 0 else 0.0
    return {
        "n": len(pred),
        "pearson_r": r,
        "log_mae": statistics.mean(log_errs),
        "samples": list(zip(truth[:6], pred[:6])),
    }


def t1_clock_consistency(model, device, n_per_tau=8):
    taus = [15, 60, 300, 1800, 7200, 43200, 86400, 604800]
    prompt = "<|im_start|>user\nHow long has it been since we started?<|im_end|>\n<|im_start|>assistant\n"
    pred = []
    truth = []
    examples = []
    for tau in taus:
        for _ in range(n_per_tau):
            resp = greedy_decode(model, prompt, tau_t=tau, device=device)
            sec = parse_duration_to_seconds(resp)
            if not (sec != sec):                                # not nan
                pred.append(sec)
                truth.append(tau)
            if len(examples) < 8:
                examples.append((tau, resp, sec))
    if len(pred) < 4:
        return {"n": len(pred), "pearson_r": float("nan"), "examples": examples,
                "pred": pred, "truth": truth}
    import statistics
    mp, mt = statistics.mean(pred), statistics.mean(truth)
    num = sum((p - mp) * (t - mt) for p, t in zip(pred, truth))
    denom = (sum((p - mp) ** 2 for p in pred) * sum((t - mt) ** 2 for t in truth)) ** 0.5
    r = num / denom if denom > 0 else 0.0
    return {"n": len(pred), "pearson_r": r, "examples": examples[:8]}


ACK_KEYWORDS = ["welcome back", "been a while", "long time", "it has been", "it's been"]


def has_ack(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in ACK_KEYWORDS)


def t2_silent_gap_ack(model, device, n=30):
    prompt = (
        "<|im_start|>user\nTell me a fun fact.<|im_end|>\n"
        "<|im_start|>assistant\nOctopuses have three hearts.<|im_end|>\n"
        "<|im_start|>user\nHi again.<|im_end|>\n<|im_start|>assistant\n"
    )
    ack_small = 0
    ack_large = 0
    examples = []
    for _ in range(n):
        small = greedy_decode(model, prompt, tau_t=10.0, device=device)
        large = greedy_decode(model, prompt, tau_t=86400.0, device=device)
        if has_ack(small): ack_small += 1
        if has_ack(large): ack_large += 1
        if len(examples) < 6:
            examples.append({"delta_small_resp": small, "delta_large_resp": large})
    return {"n": n, "ack_rate_small": ack_small / n, "ack_rate_large": ack_large / n,
            "delta": (ack_large - ack_small) / n, "examples": examples}


WEEKEND_WORDS = ["weekend", "saturday", "sunday", "fun", "relax", "off"]
WEEKDAY_WORDS = ["weekday", "monday", "tuesday", "wednesday", "thursday", "friday", "work", "busy"]


def _word_rate(text: str, words: list) -> float:
    """Binary: did the response contain ANY of these words?"""
    t = text.lower()
    return float(any(w in t for w in words))


def t3_phase_discrimination(model, device, n=20):
    prompt = "<|im_start|>user\nGood morning.<|im_end|>\n<|im_start|>assistant\n"
    weekday_wkday = []
    weekday_wkend = []
    weekend_wkday = []
    weekend_wkend = []
    examples = []
    for _ in range(n):
        # Wednesday = day 2, tau = 2*86400
        wd_resp = greedy_decode(model, prompt, tau_t=2 * 86400.0, device=device)
        # Saturday = day 5, tau = 5*86400
        we_resp = greedy_decode(model, prompt, tau_t=5 * 86400.0, device=device)
        weekday_wkday.append(_word_rate(wd_resp, WEEKDAY_WORDS))
        weekday_wkend.append(_word_rate(wd_resp, WEEKEND_WORDS))
        weekend_wkday.append(_word_rate(we_resp, WEEKDAY_WORDS))
        weekend_wkend.append(_word_rate(we_resp, WEEKEND_WORDS))
        if len(examples) < 6:
            examples.append({"wkday_resp": wd_resp, "wkend_resp": we_resp})
    import statistics
    weekday_signal = statistics.mean(weekday_wkday) - statistics.mean(weekend_wkday)
    weekend_signal = statistics.mean(weekend_wkend) - statistics.mean(weekday_wkend)
    return {
        "n": n,
        "weekday_word_rate_on_weekday_prompt": statistics.mean(weekday_wkday),
        "weekday_word_rate_on_weekend_prompt": statistics.mean(weekend_wkday),
        "weekend_word_rate_on_weekday_prompt": statistics.mean(weekday_wkend),
        "weekend_word_rate_on_weekend_prompt": statistics.mean(weekend_wkend),
        "weekday_signal": weekday_signal,
        "weekend_signal": weekend_signal,
        "examples": examples,
    }


@torch.no_grad()
def _generate_with_tau(model, prompt: str, tau_t: float, n_steps: int,
                      device: str):
    """Greedy decode n_steps tokens at fixed tau_t. Returns list of
    log-softmax distributions at each generated position."""
    tok = model.tokenizer
    prompt = _maybe_inject_prompt(prompt, tau_t)
    ids = tok.encode(prompt, return_tensors="pt").squeeze(0).to(device)
    cur = ids
    dists = []
    for _ in range(n_steps):
        out = model(cur, tau_t=tau_t)
        logits = out["logits"][-1].float().cpu()
        dists.append(F.log_softmax(logits, dim=-1))
        next_id = int(out["logits"][-1].argmax().item())
        cur = torch.cat([cur, torch.tensor([next_id], device=device)])
        if hasattr(tok, "convert_tokens_to_ids"):
            im_end = tok.convert_tokens_to_ids("<|im_end|>")
            if im_end is not None and next_id == im_end:
                break
        if next_id == tok.eos_token_id:
            break
    return dists


def t4_mutability(model, device, n_prompts=3, n_positions=8):
    """Negative control: do logits across positions differ for the
    SAME prompt across different tau values?

    v2 (2026-05-23): averages KL across the first N generated
    positions, not just the first. v1's first-position-only metric
    missed distributed chrono usage in models that route the signal
    through later semantic features (v15 case). New default
    n_positions=8 captures local + early-context chrono effects."""
    prompts = [
        "<|im_start|>user\nHello.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat time is it?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nGreetings.<|im_end|>\n<|im_start|>assistant\n",
    ][:n_prompts]
    taus = [15, 3600, 86400]
    pairwise_kls_first = []
    pairwise_kls_multi_pos = []
    per_position_kls = [[] for _ in range(n_positions)]
    for p in prompts:
        # Greedy decode at each tau, collecting distributions per position
        dist_lists = [_generate_with_tau(model, p, tau, n_positions, device)
                      for tau in taus]
        # Align position by position
        n_pos = min(len(d) for d in dist_lists)
        for pos in range(n_pos):
            ls = [dist_lists[ti][pos] for ti in range(len(taus))]
            for i in range(len(ls)):
                for j in range(i + 1, len(ls)):
                    p_a = ls[i].exp(); p_b = ls[j].exp()
                    kl = 0.5 * ((p_a * (ls[i] - ls[j])).sum() +
                                (p_b * (ls[j] - ls[i])).sum()).item()
                    per_position_kls[pos].append(kl)
                    pairwise_kls_multi_pos.append(kl)
                    if pos == 0:
                        pairwise_kls_first.append(kl)
    import statistics
    per_pos_means = [statistics.mean(p) if p else 0.0
                     for p in per_position_kls]
    return {
        "n_pairwise_first": len(pairwise_kls_first),
        "n_pairwise_multi_pos": len(pairwise_kls_multi_pos),
        "n_positions": n_positions,
        "mean_pairwise_kl": (statistics.mean(pairwise_kls_first)
                             if pairwise_kls_first else 0.0),
        "max_pairwise_kl": (max(pairwise_kls_first)
                            if pairwise_kls_first else 0.0),
        "mean_pairwise_kl_multi_pos": (statistics.mean(pairwise_kls_multi_pos)
                                       if pairwise_kls_multi_pos else 0.0),
        "max_pairwise_kl_multi_pos": (max(pairwise_kls_multi_pos)
                                      if pairwise_kls_multi_pos else 0.0),
        "per_position_mean_kls": per_pos_means,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", "--ckpt", dest="checkpoint", type=str, default=None)
    p.add_argument("--tag", type=str, default=None,
                   help="Deprecated no-op kept for old launcher compatibility.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--run-id", type=str, default=None,
                   help="Write report under runs/<run-id>/reports/ when --out is omitted.")
    p.add_argument("--base", type=str, default=V15_BASE_MODEL_NAME)
    p.add_argument("--timescales", type=str, default="",
                   help="Comma-separated chrono timescales in seconds.")
    p.add_argument("--inject-layers", type=str, default="",
                   help="Layer indices for chrono injection (must match training).")
    p.add_argument("--injection-type", type=str, default=V15_INJECTION_TYPE,
                   choices=["film", "additive"])
    p.add_argument("--inject-prompt", action="store_true",
                   help="Prepend `[elapsed: X]` to every test prompt. "
                        "Use this to eval prompt-baseline checkpoints "
                        "(trained via qwen_time_data_prompt). For CI/"
                        "additive ckpts, leave OFF.")
    p.add_argument("--lora-rank", type=int, default=V15_LORA_RANK,
                   help="LoRA adapter rank. Must match training rank.")
    p.add_argument("--use-ia3", action="store_true",
                   help="Eval IA3-trained ckpt. Must match training.")
    p.add_argument("--allow-unregistered-legacy-ckpt", action="store_true",
                   help="Allow tensor-only loading for old checkpoints that lack embedded metadata "
                        "and are not listed in checkpoints/metadata_registry.json.")
    p.add_argument("--prompt-format", type=str, default="elapsed",
                   choices=["elapsed","iso","nl"],
                   help="Format for [elapsed:X]/[timestamp:Z]/natural-language "
                        "prefix when --inject-prompt is set. Must match training format.")
    args = p.parse_args()
    if args.out is None:
        if args.run_id is None:
            raise SystemExit("output scope required: pass --out or --run-id")
        args.out = str(Path("runs") / args.run_id / "reports" / "qwen_time_check.json")
    global PROMPT_FORMAT
    PROMPT_FORMAT = args.prompt_format

    global INJECT_PROMPT_TAU
    if args.inject_prompt:
        INJECT_PROMPT_TAU = True
        print("  prompt-injection mode: ON (prepending [elapsed: X] to every test prompt)")

    cfg = QwenTimeConfig()
    cfg.base_model_name = args.base
    if args.lora_rank != V15_LORA_RANK:
        cfg.lora_rank = args.lora_rank
        print(f"  Override lora_rank: {cfg.lora_rank}")
    if args.use_ia3:
        cfg.use_ia3 = True
        print(f"  Override: IA3 PEFT (W9 baseline)")
    if args.timescales:
        cfg.timescales = tuple(int(x) for x in args.timescales.split(","))
        print(f"  Override timescales: {cfg.timescales}")
    # Honor injection-type / inject-layers from CLI for ablation eval too
    if hasattr(args, "inject_layers") and args.inject_layers:
        cfg.inject_layers = tuple(int(x) for x in args.inject_layers.split(","))
    if hasattr(args, "injection_type") and args.injection_type != "film":
        cfg.injection_type = args.injection_type
    print(f"Loading QwenTime ({cfg.base_model_name})...")
    model = build_qwen_time(cfg)
    model = model.to(args.device)
    if args.checkpoint:
        print(f"Loading {args.checkpoint}...")
        load_trainable(model, args.checkpoint, args.allow_unregistered_legacy_ckpt)
    model.train(False)

    print("\n=== T1 clock consistency (in-distribution) ===")
    t0 = time.time()
    r1 = t1_clock_consistency(model, args.device)
    print(f"  pearson r = {r1['pearson_r']:.3f} (target >= 0.8, falsify <= 0.3, n={r1['n']})")
    for tau, resp, sec in r1.get("examples", [])[:5]:
        print(f"    tau={tau:>7}  parsed={sec}  resp={resp!r}")

    print("\n=== T1b clock OOD (held-out tau) -- LOAD-BEARING TEST ===")
    r1b = t1b_clock_ood(model, args.device)
    print(f"  pearson r = {r1b.get('pearson_r', float('nan')):.3f}, log-MAE = {r1b.get('log_mae', float('nan')):.3f}")
    print(f"  (target r>=0.7 AND log_mae<0.5; this is the falsification anchor)")

    print("\n=== T2 silent-gap ack ===")
    r2 = t2_silent_gap_ack(model, args.device)
    print(f"  ack_small={r2['ack_rate_small']:.2f} ack_large={r2['ack_rate_large']:.2f} delta={r2['delta']:+.2f} (target >= 0.5)")

    print("\n=== T3 phase discrimination ===")
    r3 = t3_phase_discrimination(model, args.device)
    print(f"  weekday_signal={r3['weekday_signal']:+.3f} weekend_signal={r3['weekend_signal']:+.3f} (target >= 0.3)")

    print("\n=== T4 mutability (negative control) ===")
    r4 = t4_mutability(model, args.device)
    print(f"  mean pairwise KL across tau = {r4['mean_pairwise_kl']:.4f} (target >= 0.05 = chrono reaches output)")

    summary = {
        "T1_clock_pearson_r": r1["pearson_r"],
        "T1_pass": (r1["pearson_r"] > 0.8 if r1["pearson_r"] == r1["pearson_r"] else False),
        "T1b_ood_pearson_r": r1b.get("pearson_r", float("nan")),
        "T1b_ood_log_mae": r1b.get("log_mae", float("nan")),
        "T1b_pass": (r1b.get("pearson_r", 0) >= 0.7 and r1b.get("log_mae", 99) < 0.5),
        "T2_ack_delta": r2["delta"],
        "T2_pass": r2["delta"] >= 0.5,
        "T3_weekend_signal": r3["weekend_signal"],
        "T3_weekday_signal": r3["weekday_signal"],
        "T3_pass": max(r3["weekend_signal"], r3["weekday_signal"]) >= 0.3,
        "T4_mean_pairwise_kl": r4["mean_pairwise_kl"],
        "T4_pass": r4["mean_pairwise_kl"] >= 0.05,
    }
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nwall: {time.time() - t0:.1f}s")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "t1": r1, "t1b": r1b, "t2": r2, "t3": r3, "t4": r4}, f, indent=2)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
