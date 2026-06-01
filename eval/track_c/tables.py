"""Write Track C CSV and LaTeX tables from a headline JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def latex_table(headers: list[str], rows: list[list[Any]], caption: str, label: str) -> str:
    cols = "l" + "c" * (len(headers) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\footnotesize",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(fmt(v) for v in row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def write_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headline", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    if args.out_dir is None:
        if args.run_id is None:
            raise SystemExit("output scope required: pass --out-dir or --run-id")
        args.out_dir = str(Path("runs") / args.run_id / "reports" / "track_c" / "tables")
    out_dir = Path(args.out_dir)
    with open(args.headline) as fh:
        headline = json.load(fh)

    main_headers = ["model", "condition", "acc_mean", "acc_std", "bacc_mean", "bacc_std", "state_acc_mean", "state_acc_std"]
    main_rows: list[list[Any]] = []
    for info in headline.get("aggregate", {}).values():
        if info["condition"] == "conflict":
            main_rows.append([
                info["model_tag"],
                "conflict_scalar_follow",
                info["conflict_scalar_follow"]["mean"],
                info["conflict_scalar_follow"]["std"],
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
            ])
            main_rows.append([
                info["model_tag"],
                "conflict_prompt_follow",
                info["conflict_prompt_follow"]["mean"],
                info["conflict_prompt_follow"]["std"],
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
            ])
            continue
        main_rows.append([
            info["model_tag"],
            info["condition"],
            info["acc"]["mean"],
            info["acc"]["std"],
            info["balanced_acc"]["mean"],
            info["balanced_acc"]["std"],
            info["state_acc"]["mean"],
            info["state_acc"]["std"],
        ])
    write_csv(out_dir / "table1_main_results.csv", main_headers, main_rows)
    (out_dir / "table1_main_results.tex").write_text(
        latex_table(main_headers, main_rows, "Main Track C results.", "tab:track-c-main")
    )

    split_headers = ["model", "split", "acc_mean", "acc_std", "bacc_mean", "bacc_std", "state_acc_mean", "state_acc_std"]
    split_rows: list[list[Any]] = []
    family_headers = ["family", "acc_mean", "acc_std", "bacc_mean", "bacc_std", "state_acc_mean", "state_acc_std"]
    family_rows: list[list[Any]] = []
    for info in headline.get("aggregate_by_split", {}).values():
        split_rows.append([
            info["model_tag"],
            info["split"],
            info["acc"]["mean"],
            info["acc"]["std"],
            info["balanced_acc"]["mean"],
            info["balanced_acc"]["std"],
            info["state_acc"]["mean"],
            info["state_acc"]["std"],
        ])
    for info in headline.get("aggregate_by_family", {}).values():
        family_rows.append([
            info["family"],
            info["acc"]["mean"],
            info["acc"]["std"],
            info["balanced_acc"]["mean"],
            info["balanced_acc"]["std"],
            info["state_acc"]["mean"],
            info["state_acc"]["std"],
        ])
    write_csv(out_dir / "table2_split_generalization.csv", split_headers, split_rows)
    (out_dir / "table2_split_generalization.tex").write_text(
        latex_table(split_headers, split_rows, "Track C split generalization.", "tab:track-c-splits")
    )
    write_csv(out_dir / "table3_family_results.csv", family_headers, family_rows)
    (out_dir / "table3_family_results.tex").write_text(
        latex_table(family_headers, family_rows, "Track C family-level results.", "tab:track-c-family")
    )

    baseline_headers = ["baseline", "standard_test_acc", "heldout_template_acc", "heldout_duration_acc", "heldout_composition_acc", "heldout_family_acc"]
    baseline_values: dict[str, dict[str, list[float]]] = {}
    for baseline_payload in headline.get("baselines", {}).values():
        for name, info in baseline_payload.get("baselines", {}).items():
            baseline_values.setdefault(name, {})
            by_split = info.get("by_split", {})
            for split in ("standard_test", "heldout_template", "heldout_duration", "heldout_composition", "heldout_family"):
                baseline_values[name].setdefault(split, []).append(by_split.get(split, {}).get("acc", float("nan")))

    def mean(vals: list[float]) -> float:
        clean = [v for v in vals if v == v]
        return sum(clean) / len(clean) if clean else float("nan")

    baseline_rows: list[list[Any]] = [
        [
            name,
            mean(splits.get("standard_test", [])),
            mean(splits.get("heldout_template", [])),
            mean(splits.get("heldout_duration", [])),
            mean(splits.get("heldout_composition", [])),
            mean(splits.get("heldout_family", [])),
        ]
        for name, splits in sorted(baseline_values.items())
    ]
    write_csv(out_dir / "table4_linear_baselines.csv", baseline_headers, baseline_rows)
    (out_dir / "table4_linear_baselines.tex").write_text(
        latex_table(baseline_headers, baseline_rows, "Track C diagnostic baselines.", "tab:track-c-baselines")
    )

    shuffle_headers = ["seed", "acc_ci_hidden", "acc_shuffled", "delta_shuffle", "wrong_state_consistency"]
    shuffle_rows = [
        [
            row["seed"],
            row["acc_ci_hidden"],
            row["acc_shuffled"],
            row["delta_shuffle"],
            row["wrong_state_consistency"],
        ]
        for row in headline.get("shuffled_analysis", [])
    ]
    write_csv(out_dir / "table5_shuffled_time.csv", shuffle_headers, shuffle_rows)
    (out_dir / "table5_shuffled_time.tex").write_text(
        latex_table(shuffle_headers, shuffle_rows, "Track C shuffled-time causal analysis.", "tab:track-c-shuffle")
    )
    print(f"wrote Track C tables to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
