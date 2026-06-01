"""Diagnostic non-LLM baselines for Track C."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression

from eval.track_c.generate import label_for_state, predicate_signature_for_action
from eval.track_c.metrics import accuracy, balanced_accuracy, state_accuracy
from eval.track_c.schemas import ACTIONS, FAMILIES, TIMESCALES


ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTIONS)}
INDEX_TO_ACTION = {idx: action for action, idx in ACTION_TO_INDEX.items()}
RISK_TO_INDEX = {"low": 0, "medium": 1, "high": 2}
EVAL_SPLITS = (
    "standard_test",
    "heldout_template",
    "heldout_duration",
    "heldout_composition",
    "heldout_family",
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def chi(tau: float) -> np.ndarray:
    feats: list[float] = []
    for scale in TIMESCALES:
        feats.append(math.sin(tau / scale))
        feats.append(math.cos(tau / scale))
    feats.append(math.log1p(tau))
    return np.array(feats, dtype=np.float64)


def state_features(row: dict[str, Any]) -> np.ndarray:
    state = row["hidden_state_json"]
    feats: list[float] = []
    for name in FAMILIES:
        feats.append(1.0 if row["family"] == name else 0.0)
    for key in ("d_cache", "d_job", "d_deadline", "p_quota", "w_quota", "d_stale"):
        feats.append(math.log1p(float(state.get(key, 0.0))))
    risk = state.get("r_risk")
    for name in ("low", "medium", "high"):
        feats.append(1.0 if risk == name else 0.0)
    for key in ("cache", "job", "deadline", "quota", "modulo", "staleness", "risk"):
        feats.append(1.0 if key in row.get("active_constraints", []) else 0.0)
    return np.array(feats, dtype=np.float64)


def featurize(rows: list[dict[str, Any]], mode: str) -> tuple[np.ndarray, np.ndarray]:
    X: list[np.ndarray] = []
    y: list[int] = []
    for row in rows:
        if mode == "chi_tau_only":
            vec = chi(float(row["tau_seconds"]))
        elif mode == "state_only":
            vec = state_features(row)
        elif mode == "chi_tau_plus_state":
            vec = np.concatenate([chi(float(row["tau_seconds"])), state_features(row)])
        else:
            raise ValueError(f"unknown baseline feature mode {mode!r}")
        X.append(vec)
        y.append(ACTION_TO_INDEX[row["gold_action"]])
    return np.array(X), np.array(y)


def predict_rows(rows: list[dict[str, Any]], clf: LogisticRegression, mode: str) -> list[dict[str, Any]]:
    X, _ = featurize(rows, mode)
    pred_idx = clf.predict(X)
    out: list[dict[str, Any]] = []
    for row, idx in zip(rows, pred_idx):
        out.append({
            "item_id": row["id"],
            "seed": row["seed"],
            "split": row["split"],
            "family": row["family"],
            "condition": mode,
            "action": INDEX_TO_ACTION[int(idx)],
            "gold_action": row["gold_action"],
            "num_constraints": row["num_constraints"],
            "gold_predicates": row.get("gold_predicates"),
            "predicted_predicates": predicate_signature_for_action(
                row["family"],
                row["hidden_state_json"],
                int(row["tau_seconds"]),
                row.get("active_constraints") or [],
                INDEX_TO_ACTION[int(idx)],
            ),
        })
    return out


def rule_oracle_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        action, _, _, _ = label_for_state(
            row["family"],
            row["hidden_state_json"],
            int(row["tau_seconds"]),
            row.get("active_constraints") or [],
        )
        out.append({
            "item_id": row["id"],
            "seed": row["seed"],
            "split": row["split"],
            "family": row["family"],
            "condition": "rule_oracle",
            "action": action,
            "gold_action": row["gold_action"],
            "num_constraints": row["num_constraints"],
            "gold_predicates": row.get("gold_predicates"),
            "predicted_predicates": predicate_signature_for_action(
                row["family"],
                row["hidden_state_json"],
                int(row["tau_seconds"]),
                row.get("active_constraints") or [],
                action,
            ),
        })
    return out


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    acc, n = accuracy(rows)
    bacc, _ = balanced_accuracy(rows)
    sacc, _ = state_accuracy(rows)
    return {"acc": acc, "balanced_acc": bacc, "state_acc": sacc, "n": n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    if args.out is None:
        if args.run_id is None:
            raise SystemExit("output scope required: pass --out or --run-id")
        args.out = str(Path("runs") / args.run_id / "reports" / "track_c" / "linear_baselines.json")
    train_rows = list(iter_jsonl(Path(args.train)))
    eval_rows: list[dict[str, Any]] = []
    for path in args.eval:
        eval_rows.extend(iter_jsonl(Path(path)))
    payload: dict[str, Any] = {
        "track": "C",
        "train": str(Path(args.train)),
        "eval": [str(Path(p)) for p in args.eval],
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "baselines": {},
    }
    for mode in ("chi_tau_only", "state_only", "chi_tau_plus_state"):
        Xtr, ytr = featurize(train_rows, mode)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(Xtr, ytr)
        pred = predict_rows(eval_rows, clf, mode)
        by_split = {split: metric_summary([r for r in pred if r["split"] == split]) for split in EVAL_SPLITS}
        payload["baselines"][mode] = {
            "overall": metric_summary(pred),
            "by_split": by_split,
        }
    oracle_pred = rule_oracle_rows(eval_rows)
    payload["baselines"]["rule_oracle"] = {
        "overall": metric_summary(oracle_pred),
        "by_split": {split: metric_summary([r for r in oracle_pred if r["split"] == split]) for split in EVAL_SPLITS},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote Track C baselines to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
