"""Select a Track C checkpoint from validation prediction reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.track_c.metrics import split_breakdowns


def load_report(path: str) -> dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def score_report(path: str, metric: str) -> dict[str, Any]:
    payload = load_report(path)
    rows = payload.get("results", [])
    metrics = split_breakdowns(rows)
    return {
        "path": path,
        "checkpoint": payload.get("checkpoint"),
        "metric": metric,
        "score": metrics.get(metric, float("nan")),
        "acc": metrics.get("acc", float("nan")),
        "balanced_acc": metrics.get("balanced_acc", float("nan")),
        "state_acc": metrics.get("state_acc", float("nan")),
        "n": metrics.get("n", 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--metric", choices=("acc", "balanced_acc", "state_acc"), default="balanced_acc")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [score_report(path, args.metric) for path in args.inputs if Path(path).is_file()]
    rows = [row for row in rows if row["checkpoint"]]
    if not rows:
        raise SystemExit("no validation reports with checkpoint metadata")
    rows.sort(key=lambda row: (row["score"], row["acc"], str(row["checkpoint"])), reverse=True)
    selected = rows[0]
    payload = {
        "track": "C",
        "selection_metric": args.metric,
        "selected_checkpoint": selected["checkpoint"],
        "selected": selected,
        "candidates": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(selected["checkpoint"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
