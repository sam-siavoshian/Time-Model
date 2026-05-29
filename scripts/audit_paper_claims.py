#!/usr/bin/env python3
"""Randomized paper-claim audit against canonical code and report artifacts.

This is an artifact-level audit, not an experiment rerun. Method claims are
checked against code constants, and numeric claims are checked against canonical
JSON reports referenced by the current manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = ROOT / "reports" / "current" / "claims.json"
MANIFEST_PATH = ROOT / "reports" / "current" / "manifest.json"
PAPER_PATH = ROOT / "paper" / "main.tex"

SENTINEL_IDS = [
    "encoder_formula_no_2pi",
    "tau_sessions_negative",
    "tpdr_v2_headline",
    "tps_negative",
    "half_layer_flip_counts",
    "lora_only_behavior",
]


@dataclass(frozen=True)
class ClaimCheck:
    claim_id: str
    artifact_path: str
    locator: str
    expected_value: Any
    actual_value: Any
    required_patterns: tuple[str, ...] = ()
    forbidden_patterns: tuple[str, ...] = ()
    ambiguous_patterns: tuple[str, ...] = ()
    note: str = ""


@dataclass
class AuditItem:
    claim_id: str
    paper_text_match: dict[str, list[str]]
    artifact_path: str
    locator: str
    expected_value: Any
    actual_value: Any
    verdict: str
    note: str = ""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def iter_manifest_paths(obj: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {
                "canonical_report",
                "baseline_report",
                "dataset",
                "generator",
                "manifest",
                "path",
                "successor",
            } and isinstance(value, str):
                paths.append(value)
            elif key in {
                "canonical_reports",
                "raw_inputs",
                "supporting_reports",
                "behavior_reports",
                "figure_inputs",
                "generators",
                "paper_locations",
                "include",
                "evidence",
            } and isinstance(value, list):
                paths.extend(item for item in value if isinstance(item, str))
            else:
                paths.extend(iter_manifest_paths(value))
    elif isinstance(obj, list):
        for item in obj:
            paths.extend(iter_manifest_paths(item))
    return paths


def load_manifest_report_cache(manifest: dict[str, Any]) -> dict[str, Any]:
    """Load JSON artifacts explicitly named by the manifest when present."""
    cache: dict[str, Any] = {}
    for rel in sorted(set(iter_manifest_paths(manifest))):
        if "*" in rel or not rel.endswith(".json"):
            continue
        path = ROOT / rel
        if path.exists():
            cache[rel] = load_json(path)
    return cache


def jget(obj: Any, pointer: str) -> Any:
    cur = obj
    for raw_part in pointer.strip("/").split("/"):
        if raw_part == "":
            continue
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


def fnum(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def rounded_report(value: float, digits: int) -> float:
    return float(fnum(value, digits))


def collect_claim_checks(root: Path = ROOT) -> list[ClaimCheck]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from model import qwen_time

    checks: list[ClaimCheck] = []

    timescales = qwen_time.V15_TIMESCALES
    n_chrono = 2 * len(timescales) + 1
    injected_layers = qwen_time.resolve_inject_layers(qwen_time.QwenTimeConfig(), 36)

    checks.extend(
        [
            ClaimCheck(
                claim_id="architecture_base_model",
                artifact_path="model/qwen_time.py",
                locator="V15_BASE_MODEL_NAME",
                expected_value="Qwen 2.5 3B-Instruct",
                actual_value=qwen_time.V15_BASE_MODEL_NAME,
                required_patterns=(r"Qwen\s*2\.5\s*3B",),
            ),
            ClaimCheck(
                claim_id="architecture_timescales_31dim",
                artifact_path="model/qwen_time.py",
                locator="V15_TIMESCALES; _Chronometric.out_dim",
                expected_value={"n_timescales": 15, "chrono_dim": 31},
                actual_value={"n_timescales": len(timescales), "chrono_dim": n_chrono},
                required_patterns=(r"31-dimensional|31 output dimensions", r"15"),
            ),
            ClaimCheck(
                claim_id="encoder_formula_no_2pi",
                artifact_path="model/qwen_time.py",
                locator="CHRONO_FORMULA; _Chronometric.forward",
                expected_value="sin(tau / T), cos(tau / T), log1p(tau); no 2*pi multiplier",
                actual_value=qwen_time.CHRONO_FORMULA,
                required_patterns=(r"sin\(\s*\\Tau/T_", r"without a \$2\\pi\$ multiplier"),
                forbidden_patterns=(r"\\omega_k\s*=\s*2\\pi/T_k", r"omega_k\s*=\s*2pi/T_k"),
            ),
            ClaimCheck(
                claim_id="architecture_injected_layers",
                artifact_path="model/qwen_time.py",
                locator="resolve_inject_layers(QwenTimeConfig(), 36)",
                expected_value="35 of 36 decoder layers",
                actual_value={"n_layers": 36, "n_injected": len(injected_layers), "max_layer": max(injected_layers)},
                required_patterns=(r"35 of 36",),
            ),
            ClaimCheck(
                claim_id="architecture_lora_rank",
                artifact_path="model/qwen_time.py",
                locator="V15_LORA_RANK",
                expected_value=8,
                actual_value=qwen_time.V15_LORA_RANK,
                required_patterns=(r"rank-8|rank 8",),
            ),
        ]
    )

    probe = load_json(root / "reports" / "probe_per_layer_v15s_s0.json")
    trained = probe["condition_A_trained"]
    alpha_off = probe["condition_B_alpha_off"]
    min_post_l1 = min(float(v) for k, v in trained.items() if int(k) >= 1)
    checks.append(
        ClaimCheck(
            claim_id="mechanistic_probe",
            artifact_path="reports/probe_per_layer_v15s_s0.json",
            locator="/condition_A_trained and /condition_B_alpha_off",
            expected_value={
                "L0": -0.005,
                "L1": 0.99990,
                "min_L1_to_L36": ">=0.9995",
                "alpha_off": "chance",
            },
            actual_value={
                "L0": trained["0"],
                "L1": trained["1"],
                "min_L1_to_L36": min_post_l1,
                "alpha_off_L1": alpha_off["1"],
            },
            required_patterns=(r"-0\.005", r"0\.99990", r"0\.9995", r"zero-gate chance|FiLM gates are zeroed"),
        )
    )

    heldout_values = []
    for seed in range(3):
        held = load_json(root / f"reports/probe_per_layer_clock_heldout_s{seed}.json")
        heldout_values.append(float(held["condition_A_trained"]["1"]))
    checks.append(
        ClaimCheck(
            claim_id="clock_heldout_probe",
            artifact_path="reports/probe_per_layer_clock_heldout_s{0,1,2}.json",
            locator="/condition_A_trained/1",
            expected_value="0.99984 +/- 0.00007",
            actual_value={
                "values": heldout_values,
                "mean": statistics.mean(heldout_values),
                "sample_std": statistics.stdev(heldout_values),
            },
            required_patterns=(r"0\.99984\\pm0\.00007",),
        )
    )

    ext_ci = load_json(root / "reports" / "ext_bench_ci.json")
    ext_prompt = load_json(root / "reports" / "ext_bench_prompt.json")
    ext_vanilla = load_json(root / "reports" / "ext_bench_vanilla.json")
    checks.append(
        ClaimCheck(
            claim_id="tau_sessions_negative",
            artifact_path="reports/ext_bench_{vanilla,prompt,ci}.json",
            locator="/summary/composite_score and /summary/per_task/*",
            expected_value="prompt beats CI on composite, staleness, duration log-MAE, and adaptive r",
            actual_value={
                "ci_composite": ext_ci["summary"]["composite_score"],
                "prompt_composite": ext_prompt["summary"]["composite_score"],
                "vanilla_composite": ext_vanilla["summary"]["composite_score"],
                "ci_adaptive_r": ext_ci["summary"]["per_task"]["adaptive"]["pearson_r"],
                "prompt_adaptive_r": ext_prompt["summary"]["per_task"]["adaptive"]["pearson_r"],
                "ci_staleness": ext_ci["summary"]["per_task"]["staleness"]["accuracy"],
                "prompt_staleness": ext_prompt["summary"]["per_task"]["staleness"]["accuracy"],
                "ci_duration_log_mae": ext_ci["summary"]["per_task"]["duration_recall"]["log_mae"],
                "prompt_duration_log_mae": ext_prompt["summary"]["per_task"]["duration_recall"]["log_mae"],
            },
            required_patterns=(r"0\.377", r"0\.520", r"0\.23", r"0\.51", r"0\.098", r"0\.064", r"-0\.050", r"\+0\.115"),
            forbidden_patterns=(r"\+0\.184", r"r\s*=\s*0\.184"),
        )
    )

    tpdr = load_json(root / "reports" / "tpdr_v2_headline.json")
    tpdr_primary = tpdr["pairs"]["seed0"]["metrics"]["deliberative_score"]
    checks.append(
        ClaimCheck(
            claim_id="tpdr_v2_headline",
            artifact_path="reports/tpdr_v2_headline.json",
            locator="/pairs/seed0/metrics/deliberative_score",
            expected_value="v2 headline confirms; cross-seed unstable",
            actual_value={
                "t": tpdr_primary["t"],
                "df": tpdr_primary["df"],
                "p_one": tpdr_primary["p_one"],
                "n_paired": tpdr_primary["n_paired"],
                "mean_diff": tpdr_primary["mean_diff"],
                "seed1_p_one": tpdr["pairs"]["seed1"]["metrics"]["deliberative_score"]["p_one"],
                "seed2_t": tpdr["pairs"]["seed2"]["metrics"]["deliberative_score"]["t"],
            },
            required_patterns=(r"2\.633\\times10\^\{-6\}|2\.633e-06", r"seed-pair-dependent|Cross-seed TPDR is not stable|cross-seed.*unstable"),
            forbidden_patterns=(r"2\.91\\times10\^\{-3\}", r"2\.91e-3", r"0\.002912"),
        )
    )

    tps = load_json(root / "reports" / "tps" / "headline.json")
    checks.append(
        ClaimCheck(
            claim_id="tps_negative",
            artifact_path="reports/tps/headline.json",
            locator="/ci_v15s_crossseed and /per_adapter/vanilla/metrics/by_condition/hidden_only",
            expected_value="CI v15s hidden-only below vanilla; monotonicity negative",
            actual_value={
                "ci_hidden_only_mean": tps["ci_v15s_crossseed"]["hidden_only_mean"],
                "ci_hidden_only_std": tps["ci_v15s_crossseed"]["hidden_only_std"],
                "vanilla_hidden_only": tps["per_adapter"]["vanilla"]["metrics"]["by_condition"]["hidden_only"]["policy_acc"],
                "monotonicity_mean": tps["ci_v15s_crossseed"]["monotonicity_mean"],
            },
            required_patterns=(r"0\.344\\pm0\.005", r"vanilla 0\.349", r"r=-0\.270"),
        )
    )

    lora_reports = [
        "reports/qwen_time_lora_only_20260523_182213_recall.json",
        "reports/qwen_time_lora_only_seeds_20260524_033704_seed1_recall.json",
        "reports/qwen_time_lora_only_seeds_20260524_033704_seed2_recall.json",
    ]
    lora_flags = []
    for rel in lora_reports:
        summary = load_json(root / rel)["summary"]
        lora_flags.append({k: summary[k] for k in sorted(summary) if k.endswith("_pass")})
    checks.append(
        ClaimCheck(
            claim_id="lora_only_behavior",
            artifact_path="reports/qwen_time_lora_only*_recall.json",
            locator="/summary/*_pass",
            expected_value="all pass-style checks are false across three seeds",
            actual_value=lora_flags,
            required_patterns=(r"LoRA-only.*pass-style behavioral checks fail|All pass-style behavioral checks fail",),
            ambiguous_patterns=(r"LoRA-only training collapses every behavioral test to \$0\.000\$", r"All behavioral tests collapse to \$0\.000\$"),
            note="T1b log-MAE is not itself a zero-valued pass/fail metric.",
        )
    )

    chrono_reports = [
        "reports/chrono_only_s0_recall.json",
        "reports/chrono_only_s1_recall.json",
        "reports/chrono_only_s2_recall.json",
    ]
    chrono_flags = []
    for rel in chrono_reports:
        summary = load_json(root / rel)["summary"]
        chrono_flags.append({k: summary[k] for k in sorted(summary) if k.endswith("_pass")})
    checks.append(
        ClaimCheck(
            claim_id="chrono_only_behavior",
            artifact_path="reports/chrono_only_s{0,1,2}_recall.json",
            locator="/summary/*_pass",
            expected_value="all in-house checks pass across three seeds",
            actual_value=chrono_flags,
            required_patterns=(r"chrono-only training with LoRA frozen passes|Chrono-only ablation.*Passes",),
        )
    )

    clock_reports = [
        "reports/clock_heldout_s0_recall.json",
        "reports/clock_heldout_s1_recall.json",
        "reports/clock_heldout_s2_recall.json",
    ]
    clock_flags = []
    for rel in clock_reports:
        summary = load_json(root / rel)["summary"]
        clock_flags.append({k: summary[k] for k in sorted(summary) if k.endswith("_pass")})
    checks.append(
        ClaimCheck(
            claim_id="clock_heldout_behavior",
            artifact_path="reports/clock_heldout_s{0,1,2}_recall.json",
            locator="/summary/*_pass",
            expected_value="T1/T1b fail; T2/T4 pass in all three seeds",
            actual_value=clock_flags,
            required_patterns=(r"T1/T1b fail", r"T2 and T4 pass 3/3"),
        )
    )

    disproof = load_json(root / "reports" / "disproof_20260522_224016_falsify.json")
    checks.append(
        ClaimCheck(
            claim_id="all_layer_sign_flip",
            artifact_path="reports/disproof_20260522_224016_falsify.json",
            locator="/verdict/E_alpha_flipped_r",
            expected_value="-0.9998 with small parseable-n caveat",
            actual_value=disproof["verdict"]["E_alpha_flipped_r"],
            required_patterns=(r"-0\.9998", r"Small parseable effective \$n\$|small parseable effective sample"),
        )
    )

    alpha_perm = load_json(root / "reports" / "alpha_flip_permutation_100.json")
    perm_vals = [
        trial.get("r")
        for trial in alpha_perm["seeds"]["seed0"]["permutation_test"]
    ]
    finite = [float(v) for v in perm_vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    preserve = sum(v > 0.5 for v in finite)
    invert = sum(v < -0.5 for v in finite)
    nonfinite = len(perm_vals) - len(finite)
    mixed = len(finite) - preserve - invert
    checks.append(
        ClaimCheck(
            claim_id="half_layer_flip_counts",
            artifact_path="reports/alpha_flip_permutation_100.json",
            locator="/seeds/seed0/permutation_test using |r|>0.5",
            expected_value="threshold-qualified counts",
            actual_value={
                "threshold": "|r| > 0.5",
                "preserve": preserve,
                "invert": invert,
                "mixed": mixed,
                "nonfinite": nonfinite,
            },
            required_patterns=(r"\|r\|>0\.5", r"52 preserve, 18 invert, 23 (?:are )?mixed, (?:and )?7 (?:produce )?nonfinite"),
            ambiguous_patterns=(r"Half-layer flips\s*&\s*52 preserve, 18 invert, 23 mixed, 7 nonfinite",),
        )
    )

    alpha_norms = load_json(root / "reports" / "alpha_norms_cross_seed.json")
    top8_sets = {
        seed: sorted(int(x) for x in payload["top8"])
        for seed, payload in alpha_norms.items()
        if seed.startswith("seed")
    }
    checks.append(
        ClaimCheck(
            claim_id="top8_middepth_layers",
            artifact_path="reports/alpha_norms_cross_seed.json",
            locator="/seed*/top8",
            expected_value="L20--L27 top-8 band",
            actual_value=top8_sets,
            required_patterns=(r"L20--L27|L20-L27|20--27"),
        )
    )

    return checks


def find_patterns(patterns: tuple[str, ...], paper_text: str) -> list[str]:
    found = []
    for pattern in patterns:
        if re.search(pattern, paper_text, flags=re.IGNORECASE | re.DOTALL):
            found.append(pattern)
    return found


def evaluate_claim(check: ClaimCheck, paper_text: str) -> AuditItem:
    required_found = find_patterns(check.required_patterns, paper_text)
    forbidden_found = find_patterns(check.forbidden_patterns, paper_text)
    ambiguous_found = find_patterns(check.ambiguous_patterns, paper_text)
    missing_required = [p for p in check.required_patterns if p not in required_found]

    if forbidden_found:
        verdict = "stale"
    elif ambiguous_found:
        verdict = "ambiguous"
    elif missing_required:
        verdict = "missing"
    else:
        verdict = "pass"

    return AuditItem(
        claim_id=check.claim_id,
        paper_text_match={
            "required_found": required_found,
            "required_missing": missing_required,
            "forbidden_found": forbidden_found,
            "ambiguous_found": ambiguous_found,
        },
        artifact_path=check.artifact_path,
        locator=check.locator,
        expected_value=check.expected_value,
        actual_value=check.actual_value,
        verdict=verdict,
        note=check.note,
    )


def select_checks(checks: list[ClaimCheck], *, include_all: bool, sample_size: int, seed: int) -> list[ClaimCheck]:
    if include_all:
        return checks
    by_id = {check.claim_id: check for check in checks}
    selected_ids = [claim_id for claim_id in SENTINEL_IDS if claim_id in by_id]
    remaining = [check for check in checks if check.claim_id not in selected_ids]
    rng = random.Random(seed)
    sampled = rng.sample(remaining, k=min(max(sample_size, 0), len(remaining)))
    return [by_id[claim_id] for claim_id in selected_ids] + sampled


def run_audit(
    *,
    root: Path = ROOT,
    paper_text: str | None = None,
    include_all: bool = False,
    sample_size: int = 6,
    seed: int = 0,
) -> dict[str, Any]:
    claims = load_json(root / CLAIMS_PATH.relative_to(ROOT))
    manifest = load_json(root / MANIFEST_PATH.relative_to(ROOT))
    report_cache = load_manifest_report_cache(manifest)
    text = paper_text if paper_text is not None else (root / PAPER_PATH.relative_to(ROOT)).read_text()

    checks = collect_claim_checks(root)
    selected = select_checks(checks, include_all=include_all, sample_size=sample_size, seed=seed)
    items = [evaluate_claim(check, text) for check in selected]
    verdict_counts: dict[str, int] = {}
    for item in items:
        verdict_counts[item.verdict] = verdict_counts.get(item.verdict, 0) + 1
    ok = all(item.verdict == "pass" for item in items)
    return {
        "schema_version": 1,
        "audit_type": "artifact_level_claim_accuracy",
        "source_policy": {
            "method_source": "code",
            "numeric_source": "canonical JSON reports",
            "no_model_outputs_recomputed": True,
        },
        "inputs": {
            "paper": "paper/main.tex",
            "claims": "reports/current/claims.json",
            "manifest": "reports/current/manifest.json",
            "manifest_reports_loaded": len(report_cache),
            "claims_schema_version": claims.get("schema_version"),
            "manifest_schema_version": manifest.get("schema_version"),
        },
        "selection": {
            "seed": seed,
            "sample_size": sample_size,
            "all": include_all,
            "sentinel_ids": SENTINEL_IDS,
            "audited_claim_ids": [item.claim_id for item in items],
        },
        "ok": ok,
        "verdict_counts": verdict_counts,
        "items": [item.__dict__ for item in items],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Claim Audit",
        "",
        f"- ok: `{payload['ok']}`",
        f"- audit type: `{payload['audit_type']}`",
        f"- seed: `{payload['selection']['seed']}`",
        f"- all claims: `{payload['selection']['all']}`",
        f"- verdict counts: `{payload['verdict_counts']}`",
        "",
        "| Claim | Verdict | Artifact | Locator |",
        "|---|---|---|---|",
    ]
    for item in payload["items"]:
        lines.append(
            f"| `{item['claim_id']}` | `{item['verdict']}` | `{item['artifact_path']}` | `{item['locator']}` |"
        )
    lines.append("")
    for item in payload["items"]:
        if item["verdict"] == "pass":
            continue
        lines.extend(
            [
                f"## {item['claim_id']}",
                "",
                f"- verdict: `{item['verdict']}`",
                f"- artifact: `{item['artifact_path']}`",
                f"- locator: `{item['locator']}`",
                f"- expected: `{item['expected_value']}`",
                f"- actual: `{item['actual_value']}`",
                f"- paper match: `{item['paper_text_match']}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Write outputs under runs/<run-id>/reports/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=6, help="Additional random claims beyond sentinels")
    parser.add_argument("--all", action="store_true", help="Audit every registered claim")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = ROOT / "runs" / args.run_id
    reports_dir = run_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    payload = run_audit(include_all=args.all, sample_size=args.sample_size, seed=args.seed)
    json_path = reports_dir / "claim_audit.json"
    md_path = reports_dir / "claim_audit.md"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    md_path.write_text(render_markdown(payload))
    print(f"wrote {json_path.relative_to(ROOT)}")
    print(f"wrote {md_path.relative_to(ROOT)}")
    if not payload["ok"]:
        print(f"claim audit failed: {payload['verdict_counts']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
