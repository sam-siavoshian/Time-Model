"""chi(tau)-only and rule-oracle baselines for TPS.

Two ceilings to anchor the LLM results:

  1. RULE ORACLE
     Apply the family threshold rule directly to (family, tau_ci).
     This is the perfect-knowledge ceiling. Always =1.0 on hidden_only
     and both_agree, because gold is defined by exactly this rule.

  2. CHI(TAU) LINEAR CLASSIFIER
     Train sklearn LogisticRegression(multinomial) on
     features = [chi(tau)] + [family one-hot] to predict gold action,
     using held-in templates only (templates 0..7, all families).
     Then eval on:
       - held-in templates (in-sample)
       - held-out templates (templates 8..11)
       - held-out family (market_data)
     This tests whether the benchmark is solvable from (tau, family)
     alone without seeing the prompt text. If yes, then beating the
     classifier is the bar a real model must clear.

Writes reports/tps/baselines.json.
"""

from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np  # type: ignore

from eval.tps.benchmark import (
    FAMILIES,
    FAMILY_BY_NAME,
    iter_items,
    TAU_VALUES_S,
)

V15_TIMESCALES = (
    2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 86400, 604800,
)


def chi(tau: float) -> np.ndarray:
    feats: list[float] = []
    for T in V15_TIMESCALES:
        omega = 2 * math.pi / T
        feats.append(math.sin(omega * tau))
        feats.append(math.cos(omega * tau))
    feats.append(math.log1p(tau))
    return np.array(feats, dtype=np.float64)


def family_onehot(family_name: str) -> np.ndarray:
    names = [f.name for f in FAMILIES]
    vec = np.zeros(len(names), dtype=np.float64)
    vec[names.index(family_name)] = 1.0
    return vec


ACTION_INDEX = {"REUSE": 0, "REFRESH": 1, "ASK": 2, "SUMMARIZE": 3}
INDEX_ACTION = {v: k for k, v in ACTION_INDEX.items()}


def featurize(items: list[dict], include_family: bool) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for item in items:
        chi_v = chi(float(item["tau_ci_s"] or 0))
        if include_family:
            fv = family_onehot(item["family"])
            X.append(np.concatenate([chi_v, fv]))
        else:
            X.append(chi_v)
        y.append(ACTION_INDEX[item["gold_scalar"]])
    return np.array(X), np.array(y)


def policy_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean()) if len(y_true) else float("nan")


def balanced_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = sorted(set(int(c) for c in y_true.tolist()))
    if not classes:
        return float("nan")
    per_class = []
    for c in classes:
        mask = y_true == c
        if not mask.any():
            continue
        per_class.append(float((y_pred[mask] == c).mean()))
    return sum(per_class) / len(per_class) if per_class else float("nan")


def rule_oracle(items: list[dict]) -> list[str]:
    preds: list[str] = []
    for item in items:
        fam = FAMILY_BY_NAME[item["family"]]
        tau = float(item["tau_ci_s"])
        preds.append(fam.long_action if tau >= fam.threshold_s else fam.short_action)
    return preds


def main() -> int:
    items = [i.to_dict() for i in iter_items()]
    # Train/eval splits: train on held-in templates (0..7), held-in families (excl market_data),
    # condition in {hidden_only, both_agree} so the label is well-defined by tau_ci.
    train = [
        it for it in items
        if (not it["held_out_template"])
        and (not it["held_out_family"])
        and it["condition"] in ("hidden_only", "both_agree")
    ]
    eval_in = [
        it for it in items
        if (not it["held_out_template"])
        and (not it["held_out_family"])
        and it["condition"] in ("hidden_only", "both_agree")
    ]
    eval_ho_template = [
        it for it in items
        if it["held_out_template"]
        and (not it["held_out_family"])
        and it["condition"] in ("hidden_only", "both_agree")
    ]
    eval_ho_family = [
        it for it in items
        if it["held_out_family"]
        and it["condition"] in ("hidden_only", "both_agree")
    ]

    # --- rule oracle (perfect knowledge of family threshold) ---
    oracle_acc = {
        "all": policy_acc(
            np.array([ACTION_INDEX[it["gold_scalar"]] for it in items]),
            np.array([ACTION_INDEX[a] for a in rule_oracle(items)]),
        ),
        "hidden_only": policy_acc(
            np.array([ACTION_INDEX[it["gold_scalar"]] for it in items if it["condition"] == "hidden_only"]),
            np.array([
                ACTION_INDEX[a]
                for a, it in zip(
                    rule_oracle([it for it in items if it["condition"] == "hidden_only"]),
                    [it for it in items if it["condition"] == "hidden_only"],
                )
            ]),
        ),
    }

    # --- chi(tau) classifier without family ---
    from sklearn.linear_model import LogisticRegression  # type: ignore

    Xtr, ytr = featurize(train, include_family=False)
    Xev, yev = featurize(eval_in, include_family=False)
    Xho_t, yho_t = featurize(eval_ho_template, include_family=False)
    Xho_f, yho_f = featurize(eval_ho_family, include_family=False)
    clf_tau = LogisticRegression(max_iter=2000)
    clf_tau.fit(Xtr, ytr)
    tau_only = {
        "in_sample_acc": policy_acc(ytr, clf_tau.predict(Xtr)),
        "held_in_template_acc": policy_acc(yev, clf_tau.predict(Xev)),
        "held_out_template_acc": policy_acc(yho_t, clf_tau.predict(Xho_t)),
        "held_out_family_acc": policy_acc(yho_f, clf_tau.predict(Xho_f)),
    }

    # --- chi(tau) + family classifier ---
    Xtr, ytr = featurize(train, include_family=True)
    Xev, yev = featurize(eval_in, include_family=True)
    Xho_t, yho_t = featurize(eval_ho_template, include_family=True)
    Xho_f, yho_f = featurize(eval_ho_family, include_family=True)
    clf_taufam = LogisticRegression(max_iter=2000)
    clf_taufam.fit(Xtr, ytr)
    tau_fam = {
        "in_sample_acc": policy_acc(ytr, clf_taufam.predict(Xtr)),
        "held_in_template_acc": policy_acc(yev, clf_taufam.predict(Xev)),
        "held_out_template_acc": policy_acc(yho_t, clf_taufam.predict(Xho_t)),
        "held_out_family_acc": policy_acc(yho_f, clf_taufam.predict(Xho_f)),
    }

    headline = {
        "rule_oracle": oracle_acc,
        "chi_tau_only": tau_only,
        "chi_tau_plus_family": tau_fam,
        "n_train": len(train),
        "n_held_out_template": len(eval_ho_template),
        "n_held_out_family": len(eval_ho_family),
    }
    out = "reports/tps/baselines.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(headline, fh, indent=2)
    print(json.dumps(headline, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
