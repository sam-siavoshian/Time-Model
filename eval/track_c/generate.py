"""Deterministic Track C dataset generator."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

from eval.track_c.schemas import (
    ACTION_DESCRIPTIONS,
    ACTIONS,
    ALL_TEMPLATES,
    FAMILY_WEIGHTS,
    HELDOUT_TAU_SECONDS,
    LETTERS,
    SPLIT_SIZES_FULL,
    SPLIT_SIZES_SAFE,
    SPLITS,
    TRAIN_TAU_SECONDS,
    TrackCRecord,
)


CACHE_WINDOWS_TRAIN = (300, 1800, 7200, 21600, 86400)
CACHE_WINDOWS_HELDOUT = (900, 3600, 10800, 43200)
JOB_DURATIONS_TRAIN = (300, 1800, 2700, 7200, 14400, 43200)
JOB_DURATIONS_HELDOUT = (900, 3600, 10800, 21600)
DEADLINE_JOB_DURATIONS_TRAIN = (1800, 2700, 3600, 7200, 10800)
DEADLINE_JOB_DURATIONS_HELDOUT = (900, 5400, 14400, 21600)
DEADLINES_TRAIN = (3600, 7200, 10800, 21600, 43200, 86400)
DEADLINES_HELDOUT = (5400, 14400, 28800, 172800)
QUOTA_PERIODS_TRAIN = (3600, 7200, 21600, 43200, 86400)
QUOTA_PERIODS_HELDOUT = (5400, 10800, 28800, 172800)
QUOTA_WINDOWS_TRAIN = (300, 900, 1800, 3600)
QUOTA_WINDOWS_HELDOUT = (600, 1200, 2700)
STALE_THRESHOLDS_TRAIN = (1800, 7200, 21600, 86400, 259200)
STALE_THRESHOLDS_HELDOUT = (900, 10800, 43200, 172800)
RISK_LEVELS = ("low", "medium", "high")
HELDOUT_FAMILY = "quota"
GROUP_SIZE = 2
TRAIN_WITH_QUOTA_HELDOUT_WEIGHTS = {
    "cache": 0.085,
    "job": 0.085,
    "deadline": 0.17,
    "staleness": 0.17,
    "multi": 0.49,
}
PREDICATE_KEYS = (
    "cache_valid",
    "job_complete",
    "deadline_feasible",
    "quota_available",
    "stale",
    "risk_level",
)


def format_elapsed(tau_s: int | float) -> str:
    tau_i = int(round(tau_s))
    if tau_i < 60:
        return f"{tau_i}s"
    if tau_i < 3600:
        m, s = divmod(tau_i, 60)
        return f"{m}m {s}s" if s else f"{m}m"
    if tau_i < 86400:
        h, rem = divmod(tau_i, 3600)
        m = rem // 60
        return f"{h}h {m}m" if m else f"{h}h"
    d, rem = divmod(tau_i, 86400)
    h = rem // 3600
    return f"{d}d {h}h" if h else f"{d}d"


def tau_bucket(tau_s: int) -> str:
    return format_elapsed(tau_s)


def _choose(rng: random.Random, values: Iterable[Any]) -> Any:
    return rng.choice(tuple(values))


def _weighted_family(rng: random.Random, families: tuple[str, ...]) -> str:
    if set(families) == set(TRAIN_WITH_QUOTA_HELDOUT_WEIGHTS):
        weights = [TRAIN_WITH_QUOTA_HELDOUT_WEIGHTS[f] for f in families]
    else:
        weights = [FAMILY_WEIGHTS[f] for f in families]
    return rng.choices(families, weights=weights, k=1)[0]


def _split_families(split: str) -> tuple[str, ...]:
    if split == "heldout_family":
        return (HELDOUT_FAMILY,)
    if split == "heldout_composition":
        return ("multi",)
    if split in {"train", "val", "standard_test", "heldout_template", "heldout_duration"}:
        return tuple(f for f in FAMILY_WEIGHTS if f != HELDOUT_FAMILY)
    return tuple(FAMILY_WEIGHTS)


def _template_id(rng: random.Random, split: str) -> int:
    if split == "heldout_template":
        return rng.choice(tuple(range(8, 12)))
    return rng.choice(tuple(range(0, 8)))


def _tau_values(split: str) -> tuple[int, ...]:
    if split == "heldout_duration":
        return HELDOUT_TAU_SECONDS
    return TRAIN_TAU_SECONDS


def _param_values(split: str, train_values: tuple[int, ...], heldout_values: tuple[int, ...]) -> tuple[int, ...]:
    return heldout_values if split == "heldout_duration" else train_values


def _sample_state(rng: random.Random, split: str, family: str) -> dict[str, Any]:
    state: dict[str, Any] = {}
    if family in {"cache", "multi"}:
        state["d_cache"] = _choose(rng, _param_values(split, CACHE_WINDOWS_TRAIN, CACHE_WINDOWS_HELDOUT))
    if family == "job":
        state["d_job"] = _choose(rng, _param_values(split, JOB_DURATIONS_TRAIN, JOB_DURATIONS_HELDOUT))
    if family in {"deadline", "multi"}:
        state["d_job"] = _choose(rng, _param_values(split, DEADLINE_JOB_DURATIONS_TRAIN, DEADLINE_JOB_DURATIONS_HELDOUT))
    if family in {"deadline", "multi"}:
        deadlines = _param_values(split, DEADLINES_TRAIN, DEADLINES_HELDOUT)
        # Keep the state valid while still allowing deadline misses after tau.
        state["d_deadline"] = _choose(rng, deadlines)
    if family in {"quota", "multi"}:
        state["q_available_initial"] = False
        state["p_quota"] = _choose(rng, _param_values(split, QUOTA_PERIODS_TRAIN, QUOTA_PERIODS_HELDOUT))
        state["w_quota"] = _choose(rng, _param_values(split, QUOTA_WINDOWS_TRAIN, QUOTA_WINDOWS_HELDOUT))
        if state["w_quota"] >= state["p_quota"]:
            state["w_quota"] = max(1, state["p_quota"] // 4)
    if family in {"staleness", "multi"}:
        state["d_stale"] = _choose(rng, _param_values(split, STALE_THRESHOLDS_TRAIN, STALE_THRESHOLDS_HELDOUT))
        state["r_risk"] = _choose(rng, RISK_LEVELS)
    return state


def _active_constraints(rng: random.Random, split: str, family: str) -> list[str]:
    base = {
        "cache": ["cache"],
        "job": ["job"],
        "deadline": ["job", "deadline"],
        "quota": ["quota", "modulo"],
        "staleness": ["staleness", "risk"],
    }
    if family != "multi":
        return list(base[family])
    options = ["deadline", "quota", "cache", "job"]
    if split == "heldout_composition":
        n = rng.choice((3, 4))
    else:
        n = rng.choice((1, 2))
    return sorted(rng.sample(options, n))


def update_state(family: str, state: dict[str, Any], tau: int, active_constraints: list[str] | None = None) -> dict[str, Any]:
    active = set(active_constraints or [])
    updated: dict[str, Any] = {
        "tau_seconds": int(tau),
        "cache_age": int(tau),
    }
    if family in {"cache", "multi"}:
        updated["cache_valid"] = bool(tau < int(state["d_cache"]))
    if family in {"job", "deadline", "multi"}:
        updated["job_complete"] = bool(tau >= int(state["d_job"]))
    if family in {"deadline", "multi"}:
        remaining = int(state["d_deadline"]) - tau
        updated["deadline_remaining"] = remaining
        updated["deadline_feasible"] = bool(tau + int(state["d_job"]) <= int(state["d_deadline"]))
    if family in {"quota", "multi"}:
        period = int(state["p_quota"])
        r = tau % period
        updated["quota_phase"] = r
        updated["quota_until_reset"] = period - r if r else 0
        updated["quota_available"] = bool(r <= int(state["w_quota"]))
        updated["quota_reset"] = bool(tau >= period)
    if family in {"staleness", "multi"}:
        updated["stale"] = bool(tau >= int(state["d_stale"]))
        updated["risk_level"] = state["r_risk"]
    if family == "multi":
        updated["active_constraints"] = sorted(active)
    return updated


def predicate_signature_for_action(
    family: str,
    state: dict[str, Any],
    tau: int,
    active_constraints: list[str] | None,
    action: str,
) -> dict[str, Any]:
    """Return the temporal predicate signature implied by an action.

    The signature is family-specific and deliberately excludes the action label
    itself, so state-composition accuracy can measure predicate agreement
    instead of collapsing to ordinary action accuracy.
    """
    updated = update_state(family, state, tau, active_constraints)
    active = set(active_constraints or [])
    impossible = {"impossible_action": action}

    if family == "cache":
        return {"cache_valid": True} if action == "REUSE" else {"cache_valid": False} if action == "REFRESH" else impossible
    if family == "job":
        return {"job_complete": False} if action == "WAIT" else {"job_complete": True} if action == "RETRIEVE" else impossible
    if family == "deadline":
        if action == "DEADLINE_MISSED":
            return {"deadline_feasible": False}
        if action == "WAIT":
            return {"deadline_feasible": True, "job_complete": False}
        if action == "RETRIEVE":
            return {"deadline_feasible": True, "job_complete": True}
        return impossible
    if family == "quota":
        return {"quota_available": True} if action == "RETRIEVE" else {"quota_available": False} if action == "WAIT_QUOTA" else impossible
    if family == "staleness":
        risk = state["r_risk"]
        if action == "REUSE":
            return {"stale": False, "risk_level": risk}
        if action == "REFRESH":
            return {"stale": True, "risk_level": "medium"}
        if action == "ASK_CONFIRM":
            if risk == "high":
                return {"risk_level": "high"}
            return {"stale": True, "risk_level": "low"}
        return impossible
    if family == "multi":
        sig: dict[str, Any] = {}
        if action == "DEADLINE_MISSED" and "deadline" in active:
            return {"deadline_feasible": False}
        if "deadline" in active:
            sig["deadline_feasible"] = True
        if action == "WAIT_QUOTA" and "quota" in active:
            sig["quota_available"] = False
            return sig
        if "quota" in active:
            sig["quota_available"] = True
        if action == "REFRESH" and "cache" in active:
            sig["cache_valid"] = False
            return sig
        if "cache" in active:
            sig["cache_valid"] = True
        if action == "WAIT" and "job" in active:
            sig["job_complete"] = False
            return sig
        if "job" in active:
            sig["job_complete"] = True
        if action == "RETRIEVE":
            return sig
        if action == "ASK_CONFIRM":
            sig["risk_level"] = "high"
            return sig
        return impossible
    return impossible


def _gold_predicates(
    family: str,
    state: dict[str, Any],
    tau: int,
    active_constraints: list[str] | None,
    action: str,
) -> dict[str, Any]:
    return predicate_signature_for_action(family, state, tau, active_constraints, action)


def label_for_state(
    family: str,
    state: dict[str, Any],
    tau: int,
    active_constraints: list[str] | None = None,
) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    updated = update_state(family, state, tau, active_constraints)
    active = set(active_constraints or [])
    if family == "cache":
        action = "REUSE" if updated["cache_valid"] else "REFRESH"
        rationale = f"cache_valid={updated['cache_valid']}"
    elif family == "job":
        action = "RETRIEVE" if updated["job_complete"] else "WAIT"
        rationale = f"job_complete={updated['job_complete']}"
    elif family == "deadline":
        if not updated["deadline_feasible"]:
            action = "DEADLINE_MISSED"
        elif not updated["job_complete"]:
            action = "WAIT"
        else:
            action = "RETRIEVE"
        rationale = (
            f"job_complete={updated['job_complete']}; "
            f"deadline_feasible={updated['deadline_feasible']}"
        )
    elif family == "quota":
        action = "RETRIEVE" if updated["quota_available"] else "WAIT_QUOTA"
        rationale = (
            f"tau mod p_quota={updated['quota_phase']}; "
            f"quota_available={updated['quota_available']}"
        )
    elif family == "staleness":
        risk = state["r_risk"]
        if risk == "high":
            action = "ASK_CONFIRM"
        elif not updated["stale"]:
            action = "REUSE"
        elif risk == "low":
            action = "ASK_CONFIRM"
        else:
            action = "REFRESH"
        rationale = f"stale={updated['stale']}; risk={risk}"
    elif family == "multi":
        if "deadline" in active and not updated["deadline_feasible"]:
            action = "DEADLINE_MISSED"
        elif "quota" in active and not updated["quota_available"]:
            action = "WAIT_QUOTA"
        elif "cache" in active and not updated["cache_valid"]:
            action = "REFRESH"
        elif "job" in active and not updated["job_complete"]:
            action = "WAIT"
        elif all((
            "cache" not in active or updated["cache_valid"],
            "job" not in active or updated["job_complete"],
            "deadline" not in active or updated["deadline_feasible"],
            "quota" not in active or updated["quota_available"],
        )):
            action = "RETRIEVE"
        elif state.get("r_risk") == "high":
            action = "ASK_CONFIRM"
        else:
            action = "RETRIEVE"
        rationale = "; ".join(f"{k}={updated.get(k)}" for k in sorted(updated) if k not in {"tau_seconds", "cache_age"})
    else:
        raise ValueError(f"unknown Track C family {family!r}")
    return action, updated, rationale, _gold_predicates(family, state, tau, active_constraints, action)


def render_state_text(family: str, state: dict[str, Any], active_constraints: list[str]) -> str:
    lines = ["System/task state:"]
    if family in {"cache", "multi"}:
        lines.append(f"At the start of the session, a cached result was created. The cache expires after {format_elapsed(state['d_cache'])}.")
    if family in {"job", "deadline", "multi"}:
        lines.append(f"A background job was launched at the start of the session. The job takes {format_elapsed(state['d_job'])} to finish.")
    if family in {"deadline", "multi"}:
        lines.append(f"The user's deadline is {format_elapsed(state['d_deadline'])} after the session started.")
    if family in {"quota", "multi"}:
        lines.append(
            f"The API quota resets every {format_elapsed(state['p_quota'])}; quota is available for "
            f"{format_elapsed(state['w_quota'])} immediately after a reset."
        )
    if family in {"staleness", "multi"}:
        lines.append(f"The session becomes stale after {format_elapsed(state['d_stale'])}. The risk level is {state['r_risk']}.")
    if family == "multi":
        lines.append("Active policy constraints: " + ", ".join(active_constraints) + ".")
    lines.append("Do not assume any elapsed time unless it is supplied by the system.")
    return "\n".join(lines)


def render_model_prompt(record: dict[str, Any], *, elapsed_seconds_text: int | None = None) -> str:
    state_text = record["system_state_text"]
    if elapsed_seconds_text is not None:
        state_text = (
            "System/task state:\n"
            f"Elapsed time since the session started: {format_elapsed(elapsed_seconds_text)}.\n"
            + state_text.removeprefix("System/task state:\n")
        )
    choices = record["choices"]
    choice_lines = "\n".join(f"{letter}. {choices[letter]}" for letter in LETTERS)
    return (
        f"{state_text}\n\n"
        f"User:\n{record['visible_prompt']}\n\n"
        f"Choices:\n{choice_lines}\n\n"
        "Answer with the single letter A, B, C, or D."
    )


def sample_choices(rng: random.Random, gold_action: str, gold_letter: str) -> tuple[dict[str, str], dict[str, str]]:
    distractors = [a for a in ACTIONS if a != gold_action]
    rng.shuffle(distractors)
    selected = distractors[:3]
    actions_by_letter: dict[str, str] = {}
    remaining_letters = [l for l in LETTERS if l != gold_letter]
    actions_by_letter[gold_letter] = gold_action
    for letter, action in zip(remaining_letters, selected):
        actions_by_letter[letter] = action
    choices = {letter: ACTION_DESCRIPTIONS[actions_by_letter[letter]] for letter in LETTERS}
    return choices, actions_by_letter


def sample_group_choices(
    rng: random.Random,
    gold_actions: list[str],
    preferred_letter: str,
) -> tuple[dict[str, str], dict[str, str]]:
    unique_gold = list(dict.fromkeys(gold_actions))
    if len(unique_gold) > len(LETTERS):
        raise ValueError(f"too many gold actions for four-way choice set: {unique_gold}")
    remaining_letters = [letter for letter in LETTERS if letter != preferred_letter]
    rng.shuffle(remaining_letters)
    letter_order = [preferred_letter] + remaining_letters
    actions_by_letter: dict[str, str] = {}
    for letter, action in zip(letter_order, unique_gold):
        actions_by_letter[letter] = action
    distractors = [action for action in ACTIONS if action not in unique_gold]
    rng.shuffle(distractors)
    for letter in LETTERS:
        if letter not in actions_by_letter:
            actions_by_letter[letter] = distractors.pop(0)
    choices = {letter: ACTION_DESCRIPTIONS[actions_by_letter[letter]] for letter in LETTERS}
    return choices, actions_by_letter


def _letter_for_action(choice_actions: dict[str, str], action: str) -> str:
    for letter, candidate in choice_actions.items():
        if candidate == action:
            return letter
    raise ValueError(f"gold action {action!r} missing from choices {choice_actions}")


def _tau_group_for_state(
    rng: random.Random,
    tau_values: tuple[int, ...],
    family: str,
    state: dict[str, Any],
    active: list[str],
    group_size: int,
) -> list[int]:
    if group_size <= 1 or len(tau_values) == 1:
        return [_choose(rng, tau_values)]
    best = list(rng.sample(tau_values, k=min(group_size, len(tau_values))))
    for _ in range(48):
        candidate = list(rng.sample(tau_values, k=min(group_size, len(tau_values))))
        actions = {
            label_for_state(family, state, int(tau), active)[0]
            for tau in candidate
        }
        if len(actions) > 1:
            return candidate
        best = candidate
    return best


def build_split_records(seed: int, split: str, n: int) -> list[TrackCRecord]:
    rng = random.Random((seed + 13) * 1000003 + sum(ord(c) for c in split))
    families = _split_families(split)
    tau_values = _tau_values(split)
    rows: list[TrackCRecord] = []
    letter_counts = {letter: 0 for letter in LETTERS}
    group_idx = 0
    while len(rows) < n:
        family = _weighted_family(rng, families)
        template_id = _template_id(rng, split)
        state = _sample_state(rng, split, family)
        active = _active_constraints(rng, split, family)
        group_size = min(GROUP_SIZE, n - len(rows))
        taus = _tau_group_for_state(rng, tau_values, family, state, active, group_size)
        labels = [label_for_state(family, state, int(tau), active) for tau in taus]
        preferred_letter = LETTERS[group_idx % len(LETTERS)]
        choices, choice_actions = sample_group_choices(rng, [label[0] for label in labels], preferred_letter)
        visible_prompt = ALL_TEMPLATES[template_id]
        system_state_text = render_state_text(family, state, active)
        for tau, (action, updated, rationale, predicates) in zip(taus, labels):
            if len(rows) >= n:
                break
            tau_id = tau_values.index(tau)
            gold_letter = _letter_for_action(choice_actions, action)
            record_id = f"track_c_{seed}_{split}_{family}_{template_id}_{group_idx}_{tau_id}"
            rows.append(TrackCRecord(
                id=record_id,
                track="C",
                family=family,
                template_id=template_id,
                state_id=group_idx,
                tau_seconds=int(tau),
                tau_bucket=tau_bucket(tau),
                visible_prompt=visible_prompt,
                system_state_text=system_state_text,
                hidden_state_json=dict(state),
                updated_state_json=updated,
                choices=dict(choices),
                choice_actions=dict(choice_actions),
                gold_letter=gold_letter,
                gold_action=action,
                gold_rationale_symbolic=rationale,
                num_constraints=len(active),
                requires_modulo="quota" in active or "modulo" in active,
                requires_deadline_comparison="deadline" in active,
                requires_cache_check="cache" in active,
                requires_job_check="job" in active,
                split=split,
                condition="hidden_only",
                seed=seed,
                group_id=group_idx,
                tau_id=tau_id,
                heldout_template=split == "heldout_template",
                heldout_duration=split == "heldout_duration",
                heldout_composition=split == "heldout_composition",
                heldout_family=split == "heldout_family",
                active_constraints=list(active),
                gold_predicates=predicates,
            ))
            letter_counts[gold_letter] += 1
        group_idx += 1
    max_share = max(letter_counts.values()) / max(1, sum(letter_counts.values()))
    if n >= 200 and max_share > 0.30:
        raise ValueError(f"answer-letter imbalance in {split}: {letter_counts}")
    return rows


def split_sizes(profile: str) -> dict[str, int]:
    if profile == "full":
        return dict(SPLIT_SIZES_FULL)
    if profile == "safe":
        return dict(SPLIT_SIZES_SAFE)
    raise ValueError(f"unknown Track C size profile {profile!r}")


def write_jsonl(path: Path, records: Iterable[TrackCRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_dict()) + "\n")
            n += 1
    return n


def output_dir_from_args(args: argparse.Namespace) -> Path:
    if args.out_dir:
        return Path(args.out_dir)
    if args.run_id:
        return Path("runs") / args.run_id / "data" / "track_c"
    raise SystemExit("output scope required: pass --out-dir or --run-id")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", choices=("full", "safe"), default="full")
    ap.add_argument("--split", choices=SPLITS + ("all",), default="all")
    args = ap.parse_args()
    out_dir = output_dir_from_args(args)
    sizes = split_sizes(args.profile)
    splits = SPLITS if args.split == "all" else (args.split,)
    summary: dict[str, int] = {}
    for split in splits:
        rows = build_split_records(args.seed, split, sizes[split])
        out = out_dir / f"track_c_seed{args.seed}_{split}.jsonl"
        summary[split] = write_jsonl(out, rows)
        print(f"wrote {summary[split]} Track C {split} records to {out}")
    print(json.dumps({"seed": args.seed, "profile": args.profile, "splits": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
