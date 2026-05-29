"""TPS benchmark item generator.

Six policy families, each with its own freshness threshold (so a single
"tau > X => refresh" rule cannot solve the benchmark). Forced-choice
output over A/B/C/D = REUSE/REFRESH/ASK/SUMMARIZE.

Conditions:
  hidden_only       no elapsed text in prompt; tau supplied via CI scalar
  prompt_only       elapsed text in prompt; CI scalar set to 0
  both_agree        elapsed text in prompt AND CI scalar matches
  conflict_ps_cl    prompt says SHORT, CI says LONG
  conflict_pl_cs    prompt says LONG, CI says SHORT

Item gold action depends on the EFFECTIVE tau for that condition:
  hidden_only / both_agree     gold from tau_ci
  prompt_only                  gold from tau_prompt
  conflict_*                   two golds reported: scalar_follow vs prompt_follow

Splits (deterministic by item_id hash):
  train/probe = templates 0..7 in each family
  heldout_prompts = templates 8..11
  heldout_family = market_data entirely (kept out of any training-style fit)

This file generates the items only. The scoring rule oracle is encoded
here too for ceiling computation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Iterator


TAU_VALUES_S: tuple[int, ...] = (
    10,           # 10s
    30,           # 30s
    300,          # 5m
    1800,         # 30m
    7200,         # 2h
    43200,        # 12h
    86400,        # 1d
    259200,       # 3d
    604800,       # 7d
)

CONDITIONS: tuple[str, ...] = (
    "hidden_only",
    "prompt_only",
    "both_agree",
    "conflict_ps_cl",
    "conflict_pl_cs",
)

ACTIONS: tuple[str, ...] = ("REUSE", "REFRESH", "ASK", "SUMMARIZE")
ACTION_LETTERS: dict[str, str] = {"A": "REUSE", "B": "REFRESH", "C": "ASK", "D": "SUMMARIZE"}
ACTION_TO_LETTER: dict[str, str] = {v: k for k, v in ACTION_LETTERS.items()}


@dataclass(frozen=True)
class Family:
    name: str
    threshold_s: int           # tau >= threshold_s => long-side action
    short_action: str          # gold action when tau < threshold
    long_action: str           # gold action when tau >= threshold
    templates: tuple[str, ...] # base prompt texts (no time mention)


FAMILIES: tuple[Family, ...] = (
    Family(
        name="cache_reuse",
        threshold_s=3600,                # 1h
        short_action="REUSE",
        long_action="REFRESH",
        templates=(
            "Should I reuse the cached query result for the user's dashboard widget?",
            "The previous lookup returned a value. Reuse it for this rendering pass?",
            "We have a cached response for this exact request. Use cached?",
            "There is a stored result for the user's filter. Apply it or recompute?",
            "Background job already ran this aggregation. Use prior output?",
            "Memo cache contains this key. Skip recomputation?",
            "Stored answer exists for this lookup. Return cached version?",
            "Prior cache hit for this user. Serve cached payload?",
            "Cache layer has this query. Return without recompute?",
            "Result cached during last session. Reuse for current request?",
            "We cached this DB query earlier. Use it again?",
            "Cached chart data exists. Render from cache?",
        ),
    ),
    Family(
        name="session_continuation",
        threshold_s=300,                 # 5m
        short_action="REUSE",
        long_action="SUMMARIZE",
        templates=(
            "Can you continue from the previous result?",
            "Continue the conversation from where we left off.",
            "Pick up where the last turn ended.",
            "Resume the prior task without re-stating context.",
            "Continue working on the same problem from earlier.",
            "Carry on from the prior assistant response.",
            "Continue analysis on the data we were just looking at.",
            "Keep going from the last suggestion you made.",
            "Continue the agent loop from the prior step.",
            "Resume drafting the document we were writing.",
            "Continue executing the plan we discussed.",
            "Keep iterating on the same code we had open.",
        ),
    ),
    Family(
        name="api_freshness",
        threshold_s=30,                  # 30s
        short_action="REUSE",
        long_action="REFRESH",
        templates=(
            "The streaming API returned this snapshot. Use it directly?",
            "Realtime feed delivered this tick. Apply without re-fetch?",
            "Server-sent event payload received. Forward to client?",
            "Pub/sub message just arrived. Process immediately or re-query?",
            "Websocket frame contains the latest update. Trust it?",
            "Streaming endpoint sent this value. Render it?",
            "Live channel posted a result. Forward without refetch?",
            "Push notification carries the value. Use directly?",
            "Realtime broker delivered this. Treat as authoritative?",
            "Live stream returned this datum. Use unchanged?",
            "Streaming subscription pushed this. Forward?",
            "Hot feed sent the value. Apply directly?",
        ),
    ),
    Family(
        name="safety_advice",
        threshold_s=7200,                # 2h
        short_action="REUSE",
        long_action="ASK",
        templates=(
            "Should the agent proceed with the medication-dose plan from earlier?",
            "Continue the user's emergency-response workflow with prior context?",
            "Apply the prior safety-checklist outcome to this new action?",
            "Use the earlier hazard assessment for this maintenance task?",
            "Proceed with the user's allergy-disclosed prior assumptions?",
            "Continue the contraindication review from earlier session state?",
            "Apply the prior incident-triage decision to a new alert?",
            "Use earlier safety-context flags for the current intervention?",
            "Continue with the user's previously stated medical history?",
            "Apply earlier risk-tier decision to this new request?",
            "Reuse the operating-room checklist from the previous turn?",
            "Continue the patient handoff with prior summary intact?",
        ),
    ),
    Family(
        name="calendar_planning",
        threshold_s=86400,               # 1d
        short_action="REUSE",
        long_action="REFRESH",
        templates=(
            "Use the calendar snapshot we pulled earlier for scheduling?",
            "Reuse the prior availability map for this meeting placement?",
            "Apply the earlier schedule view to plan this event?",
            "Use the prior calendar export for conflict detection?",
            "Reuse the saved free/busy data for next-step routing?",
            "Apply prior availability to the proposed time?",
            "Use the cached calendar state for routing this invite?",
            "Reuse the schedule we fetched earlier?",
            "Apply prior calendar context to the booking?",
            "Use the earlier availability scan for placement?",
            "Reuse the prior week-view for the planning step?",
            "Apply the saved calendar context for slot selection?",
        ),
    ),
    Family(
        name="market_data",
        threshold_s=10,                  # 10s -- very tight (held-out family)
        short_action="REUSE",
        long_action="REFRESH",
        templates=(
            "Use the last market tick for the order-routing decision?",
            "Apply the prior quote to size this trade?",
            "Use the cached price for the execution check?",
            "Reuse the order-book snapshot for spread calc?",
            "Apply prior bid/ask to route this child order?",
            "Use cached implied-vol value to price this option?",
            "Reuse the prior NBBO for execution decisioning?",
            "Use the saved tick for this risk check?",
            "Apply the cached quote to the slippage check?",
            "Use the prior level-1 snapshot to size?",
            "Reuse the cached spread for routing?",
            "Apply the saved last-trade price?",
        ),
    ),
)


FAMILY_BY_NAME = {f.name: f for f in FAMILIES}


def gold_action(family_name: str, tau_seconds: float) -> str:
    f = FAMILY_BY_NAME[family_name]
    return f.long_action if tau_seconds >= f.threshold_s else f.short_action


def format_elapsed(tau_s: float) -> str:
    if tau_s < 60:
        return f"{int(round(tau_s))}s"
    if tau_s < 3600:
        m, s = divmod(int(round(tau_s)), 60)
        return f"{m}m {s}s" if s else f"{m}m"
    if tau_s < 86400:
        h, rem = divmod(int(round(tau_s)), 3600)
        m = rem // 60
        return f"{h}h {m}m" if m else f"{h}h"
    d, rem = divmod(int(round(tau_s)), 86400)
    h = rem // 3600
    return f"{d}d {h}h" if h else f"{d}d"


PROMPT_HEAD = (
    "You are an agent that must choose one action. Pick exactly ONE letter.\n\n"
)
PROMPT_TAIL = (
    "\n\nOptions:\n"
    "A. REUSE the prior result\n"
    "B. REFRESH from source\n"
    "C. ASK the user for confirmation\n"
    "D. SUMMARIZE prior context and revalidate\n\n"
    "Answer with the single letter A, B, C, or D."
)


def render_prompt(
    template: str,
    *,
    elapsed_text: str | None = None,
) -> str:
    """Render the user-visible prompt.

    elapsed_text=None  => no time string in the prompt (hidden_only / no-CI baselines)
    elapsed_text="3d"  => time included in prompt body
    """
    body = template
    if elapsed_text is not None:
        body = f"[elapsed: {elapsed_text}] {body}"
    return PROMPT_HEAD + body + PROMPT_TAIL


def item_id(family: str, template_idx: int, tau_ci: int, tau_prompt: int | None, condition: str) -> str:
    raw = f"{family}|{template_idx}|{tau_ci}|{tau_prompt}|{condition}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def held_out_template(template_idx: int) -> bool:
    """Templates 0..7 are held-in (training-style); 8..11 are held-out prompts."""
    return template_idx >= 8


def held_out_family(family_name: str) -> bool:
    """market_data is the entirely held-out family (Omar's compositional split)."""
    return family_name == "market_data"


@dataclass
class Item:
    item_id: str
    family: str
    template_idx: int
    template_text: str
    tau_ci_s: int               # what CI/chrono channel sees (0 if no CI)
    tau_prompt_s: int | None    # what prompt text says (None if no prompt time)
    condition: str
    prompt: str                 # fully-rendered user message
    gold_scalar: str            # action implied by tau_ci against family threshold
    gold_prompt: str | None     # action implied by tau_prompt (None if no prompt time)
    held_out_template: bool
    held_out_family: bool

    def to_dict(self) -> dict:
        return asdict(self)


def iter_items() -> Iterator[Item]:
    for family in FAMILIES:
        for t_idx, template_text in enumerate(family.templates):
            for tau_ci in TAU_VALUES_S:
                for cond in CONDITIONS:
                    if cond == "hidden_only":
                        tau_prompt: int | None = None
                    elif cond == "prompt_only":
                        # Symmetric: same tau lives in prompt text only (CI=0).
                        tau_prompt = tau_ci
                    elif cond == "both_agree":
                        tau_prompt = tau_ci
                    elif cond == "conflict_ps_cl":
                        # prompt says SHORT, CI says LONG.
                        if tau_ci < family.threshold_s:
                            continue  # need tau_ci to be on long side
                        tau_prompt = min(TAU_VALUES_S)            # 10s (short)
                    elif cond == "conflict_pl_cs":
                        if tau_ci >= family.threshold_s:
                            continue  # need tau_ci to be on short side
                        tau_prompt = max(TAU_VALUES_S)            # 7d (long)
                    else:
                        raise ValueError(f"unknown condition {cond!r}")

                    elapsed_text = None if tau_prompt is None else format_elapsed(tau_prompt)
                    if cond == "hidden_only":
                        tau_ci_effective = tau_ci
                    elif cond == "prompt_only":
                        tau_ci_effective = 0
                    else:
                        tau_ci_effective = tau_ci

                    prompt_text = render_prompt(template_text, elapsed_text=elapsed_text)
                    gold_scalar = gold_action(family.name, tau_ci_effective if cond != "prompt_only" else tau_prompt)
                    gold_prompt_val = (
                        gold_action(family.name, tau_prompt) if tau_prompt is not None else None
                    )
                    iid = item_id(family.name, t_idx, tau_ci_effective, tau_prompt, cond)
                    yield Item(
                        item_id=iid,
                        family=family.name,
                        template_idx=t_idx,
                        template_text=template_text,
                        tau_ci_s=tau_ci_effective,
                        tau_prompt_s=tau_prompt,
                        condition=cond,
                        prompt=prompt_text,
                        gold_scalar=gold_scalar,
                        gold_prompt=gold_prompt_val,
                        held_out_template=held_out_template(t_idx),
                        held_out_family=held_out_family(family.name),
                    )


def write_items(path: str) -> int:
    n = 0
    with open(path, "w") as fh:
        for item in iter_items():
            fh.write(json.dumps(item.to_dict()) + "\n")
            n += 1
    return n


if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--run-id", default=None,
                    help="Write to runs/<run-id>/data/tps/items.jsonl when --out is omitted.")
    args = ap.parse_args()
    if args.out is None:
        if args.run_id is None:
            raise SystemExit("output scope required: pass --out or --run-id")
        args.out = str(Path("runs") / args.run_id / "data" / "tps" / "items.jsonl")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = write_items(args.out)
    print(f"wrote {n} items to {args.out}")
