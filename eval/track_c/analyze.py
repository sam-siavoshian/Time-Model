"""Aggregate Track C prediction reports."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval.track_c.metrics import mean_std, split_breakdowns


SEED_RE = re.compile(r"_s(\d+)\b")


def load_report(path: str) -> dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def report_metrics(path: str) -> dict[str, Any]:
    payload = load_report(path)
    rows = payload["results"]
    metrics = split_breakdowns(rows)
    by_split = metrics.get("by_split", {})
    eval_split = next(iter(by_split), None) if len(by_split) == 1 else None
    seed = rows[0].get("seed") if rows else None
    return {
        "path": path,
        "model_tag": payload.get("model_tag", Path(path).stem),
        "model_group": normalize_model_tag(str(payload.get("model_tag", Path(path).stem))),
        "seed": seed,
        "adapter": payload.get("adapter"),
        "condition": payload.get("condition"),
        "eval_split": eval_split,
        "checkpoint": payload.get("checkpoint"),
        "items": payload.get("items"),
        "n_items": payload.get("n_items", len(rows)),
        "elapsed_sec": payload.get("elapsed_sec"),
        "metrics": metrics,
    }


def normalize_model_tag(model_tag: str) -> str:
    if model_tag.startswith("ci_track_c_s"):
        return "ci_track_c"
    if model_tag.startswith("prompt_lora_track_c_s"):
        return "prompt_lora_track_c"
    if model_tag.startswith("lora_only_track_c_s"):
        return "lora_only_track_c"
    return SEED_RE.sub("", model_tag)


def aggregate_by_model_condition(per_report: dict[str, Any], *, split: str = "standard_test") -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for info in per_report.values():
        if info.get("eval_split") != split:
            continue
        grouped[(str(info["model_group"]), str(info["condition"]))].append(info)
    out: dict[str, Any] = {}
    for (model_tag, condition), infos in grouped.items():
        key = f"{model_tag}|{condition}"
        out[key] = {
            "model_tag": model_tag,
            "condition": condition,
            "acc": mean_std([i["metrics"]["acc"] for i in infos]),
            "balanced_acc": mean_std([i["metrics"]["balanced_acc"] for i in infos]),
            "state_acc": mean_std([i["metrics"]["state_acc"] for i in infos]),
            "conflict_scalar_follow": mean_std([
                i["metrics"].get("conflict", {}).get("scalar_follow_rate", float("nan"))
                for i in infos
            ]),
            "conflict_prompt_follow": mean_std([
                i["metrics"].get("conflict", {}).get("prompt_follow_rate", float("nan"))
                for i in infos
            ]),
            "n_reports": len(infos),
        }
    return out


def aggregate_by_model_split(per_report: dict[str, Any], *, condition: str = "ci_hidden_time") -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for info in per_report.values():
        if info.get("condition") != condition or not info.get("eval_split"):
            continue
        grouped[(str(info["model_group"]), str(info["eval_split"]))].append(info)
    out: dict[str, Any] = {}
    for (model_tag, split), infos in grouped.items():
        key = f"{model_tag}|{split}"
        out[key] = {
            "model_tag": model_tag,
            "split": split,
            "acc": mean_std([i["metrics"]["acc"] for i in infos]),
            "balanced_acc": mean_std([i["metrics"]["balanced_acc"] for i in infos]),
            "state_acc": mean_std([i["metrics"]["state_acc"] for i in infos]),
            "n_reports": len(infos),
        }
    return out


def aggregate_family(per_report: dict[str, Any], *, model_group: str = "ci_track_c", condition: str = "ci_hidden_time") -> dict[str, Any]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for info in per_report.values():
        if (
            info.get("model_group") != model_group
            or info.get("condition") != condition
        ):
            continue
        eval_split = info.get("eval_split")
        for family, metrics in info["metrics"].get("by_family", {}).items():
            if family == "quota":
                if eval_split != "heldout_family":
                    continue
            elif eval_split != "standard_test":
                continue
            grouped[family].append(metrics)
    out: dict[str, Any] = {}
    for family, infos in grouped.items():
        out[family] = {
            "family": family,
            "acc": mean_std([i["acc"] for i in infos]),
            "balanced_acc": mean_std([i["balanced_acc"] for i in infos]),
            "state_acc": mean_std([i["state_acc"] for i in infos]),
            "n_reports": len(infos),
        }
    return out


def shuffled_analysis(per_report: dict[str, Any], *, model_group: str = "ci_track_c", split: str = "standard_test") -> list[dict[str, Any]]:
    by_seed: dict[int, dict[str, Any]] = defaultdict(dict)
    for info in per_report.values():
        if info.get("model_group") != model_group or info.get("eval_split") != split:
            continue
        seed = info.get("seed")
        if seed is None:
            continue
        cond = info.get("condition")
        if cond == "ci_hidden_time":
            by_seed[int(seed)]["acc_ci_hidden"] = info["metrics"]["acc"]
        elif cond == "shuffled_time_control":
            by_seed[int(seed)]["acc_shuffled"] = info["metrics"]["acc"]
            by_seed[int(seed)]["wrong_state_consistency"] = info["metrics"].get("wrong_state_consistency", {}).get("value", float("nan"))
    rows = []
    for seed, vals in sorted(by_seed.items()):
        if "acc_ci_hidden" not in vals or "acc_shuffled" not in vals:
            continue
        rows.append({
            "seed": seed,
            "acc_ci_hidden": vals["acc_ci_hidden"],
            "acc_shuffled": vals["acc_shuffled"],
            "delta_shuffle": vals["acc_ci_hidden"] - vals["acc_shuffled"],
            "wrong_state_consistency": vals.get("wrong_state_consistency", float("nan")),
        })
    return rows


def load_baselines(paths: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in paths:
        with open(path) as fh:
            out[Path(path).stem] = json.load(fh)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--inputs", nargs="+")
    src.add_argument("--input-glob")
    src.add_argument("--run-id")
    ap.add_argument("--baselines", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.inputs:
        files = args.inputs
    elif args.input_glob:
        files = sorted(glob.glob(args.input_glob))
    else:
        report_dir = root / "runs" / args.run_id / "reports" / "track_c" / "predictions"
        files = sorted(str(p) for p in report_dir.glob("*.json"))
        if args.out is None:
            args.out = str(root / "runs" / args.run_id / "reports" / "track_c" / "headline.json")
    if args.out is None:
        raise SystemExit("--out is required unless using --run-id")
    files = [str(Path(f) if Path(f).is_absolute() else root / f) for f in files]
    files = [f for f in files if Path(f).is_file()]
    if not files:
        raise SystemExit("no Track C prediction reports matched")
    per_report = {Path(path).stem: report_metrics(path) for path in files}
    headline = {
        "track": "C",
        "input_files": [
            str(Path(f).relative_to(root)) if Path(f).is_relative_to(root) else f
            for f in files
        ],
        "per_report": per_report,
        "aggregate": aggregate_by_model_condition(per_report),
        "aggregate_by_split": aggregate_by_model_split(per_report),
        "aggregate_by_family": aggregate_family(per_report),
        "shuffled_analysis": shuffled_analysis(per_report),
        "baselines": load_baselines(args.baselines),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(headline, fh, indent=2)
    print(f"wrote Track C headline to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
