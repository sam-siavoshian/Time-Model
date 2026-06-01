"""Metrics for Track C predictions and diagnostic baselines."""

from __future__ import annotations

import math
import json
from collections import defaultdict
from typing import Any, Iterable

from eval.track_c.schemas import ACTIONS


def accuracy(rows: Iterable[dict[str, Any]], *, gold_key: str = "gold_action") -> tuple[float, int]:
    vals = list(rows)
    if not vals:
        return float("nan"), 0
    return sum(1 for r in vals if r.get("action") == r.get(gold_key)) / len(vals), len(vals)


def balanced_accuracy(rows: Iterable[dict[str, Any]], *, gold_key: str = "gold_action") -> tuple[float, int]:
    vals = list(rows)
    if not vals:
        return float("nan"), 0
    by_class: dict[str, list[bool]] = defaultdict(list)
    for row in vals:
        gold = row.get(gold_key)
        if gold in ACTIONS:
            by_class[str(gold)].append(row.get("action") == gold)
    if not by_class:
        return float("nan"), len(vals)
    return sum(sum(v) / len(v) for v in by_class.values()) / len(by_class), len(vals)


def state_accuracy(rows: Iterable[dict[str, Any]], *, gold_key: str = "gold_action") -> tuple[float, int]:
    vals = list(rows)
    if not vals:
        return float("nan"), 0
    predicate_rows = [
        r for r in vals
        if r.get("gold_predicates") is not None and r.get("predicted_predicates") is not None
    ]
    if not predicate_rows:
        return accuracy(vals, gold_key=gold_key)

    def canon(obj: Any) -> str:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"))

    return (
        sum(
            1
            for row in predicate_rows
            if canon(row.get("predicted_predicates")) == canon(row.get("gold_predicates"))
        ) / len(predicate_rows),
        len(predicate_rows),
    )


def wrong_state_consistency(rows: Iterable[dict[str, Any]]) -> tuple[float, int]:
    vals = [r for r in rows if r.get("shuffled_gold_action") is not None]
    if not vals:
        return float("nan"), 0
    return sum(1 for r in vals if r.get("action") == r.get("shuffled_gold_action")) / len(vals), len(vals)


def conflict_follow(rows: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    vals = [r for r in rows if r.get("gold_prompt_action") is not None]
    if not vals:
        return {
            "n": 0,
            "scalar_follow_rate": float("nan"),
            "prompt_follow_rate": float("nan"),
        }
    scalar = sum(1 for r in vals if r.get("action") == r.get("gold_action"))
    prompt = sum(1 for r in vals if r.get("action") == r.get("gold_prompt_action"))
    return {
        "n": len(vals),
        "scalar_follow_rate": scalar / len(vals),
        "prompt_follow_rate": prompt / len(vals),
    }


def split_breakdowns(rows: list[dict[str, Any]]) -> dict[str, Any]:
    acc, n = accuracy(rows)
    bacc, _ = balanced_accuracy(rows)
    sacc, _ = state_accuracy(rows)
    out: dict[str, Any] = {
        "acc": acc,
        "balanced_acc": bacc,
        "state_acc": sacc,
        "n": n,
    }
    by_split: dict[str, Any] = {}
    for split in sorted({r["split"] for r in rows}):
        split_rows = [r for r in rows if r["split"] == split]
        sa, sn = accuracy(split_rows)
        sb, _ = balanced_accuracy(split_rows)
        ss, _ = state_accuracy(split_rows)
        by_split[split] = {"acc": sa, "balanced_acc": sb, "state_acc": ss, "n": sn}
    out["by_split"] = by_split
    by_family: dict[str, Any] = {}
    for family in sorted({r["family"] for r in rows}):
        fam_rows = [r for r in rows if r["family"] == family]
        fa, fn = accuracy(fam_rows)
        fb, _ = balanced_accuracy(fam_rows)
        fs, _ = state_accuracy(fam_rows)
        by_family[family] = {"acc": fa, "balanced_acc": fb, "state_acc": fs, "n": fn}
    out["by_family"] = by_family
    by_constraints: dict[str, Any] = {}
    for num_constraints in sorted({int(r["num_constraints"]) for r in rows}):
        c_rows = [r for r in rows if int(r["num_constraints"]) == num_constraints]
        ca, cn = accuracy(c_rows)
        cb, _ = balanced_accuracy(c_rows)
        cs, _ = state_accuracy(c_rows)
        by_constraints[str(num_constraints)] = {"acc": ca, "balanced_acc": cb, "state_acc": cs, "n": cn}
    out["by_num_constraints"] = by_constraints
    if any(r.get("shuffled_gold_action") is not None for r in rows):
        wsc, wn = wrong_state_consistency(rows)
        out["wrong_state_consistency"] = {"value": wsc, "n": wn}
    if any(r.get("gold_prompt_action") is not None for r in rows):
        out["conflict"] = conflict_follow(rows)
    return out


def mean_std(values: list[float]) -> dict[str, float | int]:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        std = 0.0
    else:
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
    return {"mean": mean, "std": std, "n": len(vals)}
