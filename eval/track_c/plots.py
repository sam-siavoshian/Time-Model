"""Generate Track C figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headline", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    if args.out_dir is None:
        if args.run_id is None:
            raise SystemExit("output scope required: pass --out-dir or --run-id")
        args.out_dir = str(Path("runs") / args.run_id / "reports" / "track_c" / "figures")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.headline) as fh:
        headline = json.load(fh)

    aggregate = headline.get("aggregate", {})
    labels = []
    values = []
    for key, info in sorted(aggregate.items()):
        if info["condition"] in {"ci_hidden_time", "no_time_control", "shuffled_time_control", "prompt_timestamp"}:
            labels.append(key.replace("|", "\n"))
            values.append(info["acc"]["mean"])
    if labels:
        plt.figure(figsize=(max(6, len(labels) * 1.2), 4))
        plt.bar(labels, values)
        plt.ylabel("Accuracy")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / "fig1_accuracy_by_condition.pdf")
        plt.close()

    by_constraint_values: dict[int, list[float]] = {}
    for info in headline.get("per_report", {}).values():
        if info.get("model_group") != "ci_track_c" or info.get("condition") != "ci_hidden_time":
            continue
        for num, metrics in info["metrics"].get("by_num_constraints", {}).items():
            by_constraint_values.setdefault(int(num), []).append(metrics["acc"])
    if by_constraint_values:
        xs = sorted(by_constraint_values)
        ys = [sum(by_constraint_values[x]) / len(by_constraint_values[x]) for x in xs]
        plt.figure(figsize=(5, 3.5))
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Number of constraints")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.savefig(out_dir / "fig2_accuracy_by_constraints.pdf")
        plt.close()

    by_family = headline.get("aggregate_by_family", {})
    if by_family:
        fams = sorted(by_family)
        vals = [by_family[f]["acc"]["mean"] for f in fams]
        plt.figure(figsize=(7, 3.5))
        plt.bar(fams, vals)
        plt.ylabel("Accuracy")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / "fig3_family_accuracy.pdf")
        plt.close()

    shuffle_rows = headline.get("shuffled_analysis", [])
    if shuffle_rows:
        hidden = sum(row["acc_ci_hidden"] for row in shuffle_rows) / len(shuffle_rows)
        shuffled = sum(row["acc_shuffled"] for row in shuffle_rows) / len(shuffle_rows)
        wsc_vals = [row["wrong_state_consistency"] for row in shuffle_rows if row["wrong_state_consistency"] == row["wrong_state_consistency"]]
        wsc = sum(wsc_vals) / len(wsc_vals) if wsc_vals else 0.0
        plt.figure(figsize=(5, 3.5))
        plt.bar(["CI", "Shuffled", "WSC"], [hidden, shuffled, wsc])
        plt.ylabel("Rate")
        plt.tight_layout()
        plt.savefig(out_dir / "fig4_shuffled_time_wsc.pdf")
        plt.close()
    print(f"wrote Track C figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
