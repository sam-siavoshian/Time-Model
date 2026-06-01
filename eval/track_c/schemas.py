"""Schemas and constants for Track C.

Track C records are intentionally plain JSON-compatible dataclasses so the
artifacts stay easy to inspect and diff. The generator may include extra helper
fields beyond the paper-facing schema, but the required fields are stable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


ACTIONS: tuple[str, ...] = (
    "REUSE",
    "WAIT",
    "RETRIEVE",
    "REFRESH",
    "WAIT_QUOTA",
    "DEADLINE_MISSED",
    "ASK_CONFIRM",
)

LETTERS: tuple[str, ...] = ("A", "B", "C", "D")

ACTION_DESCRIPTIONS: dict[str, str] = {
    "REUSE": "Reuse the current cached or prior result.",
    "WAIT": "Wait for the in-progress job to finish.",
    "RETRIEVE": "Retrieve the completed output now.",
    "REFRESH": "Refresh stale data from the source.",
    "WAIT_QUOTA": "Wait until quota is available again.",
    "DEADLINE_MISSED": "Report that the deadline can no longer be met.",
    "ASK_CONFIRM": "Ask the user to confirm before proceeding.",
}

PRIMARY_TAU_SECONDS: tuple[int, ...] = (
    10,
    30,
    300,
    1800,
    7200,
    10800,
    43200,
    86400,
    259200,
    604800,
)

TRAIN_TAU_SECONDS: tuple[int, ...] = (
    10,
    30,
    300,
    1800,
    7200,
    43200,
    86400,
    604800,
)

HELDOUT_TAU_SECONDS: tuple[int, ...] = (
    60,
    900,
    3600,
    10800,
    21600,
    172800,
    345600,
)

TIMESCALES: tuple[int, ...] = (
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    4096,
    16384,
    65536,
    86400,
    604800,
)

FAMILIES: tuple[str, ...] = (
    "cache",
    "job",
    "deadline",
    "quota",
    "staleness",
    "multi",
)

FAMILY_WEIGHTS: dict[str, float] = {
    "cache": 0.10,
    "job": 0.10,
    "deadline": 0.15,
    "quota": 0.15,
    "staleness": 0.15,
    "multi": 0.35,
}

TRAIN_TEMPLATES: tuple[str, ...] = (
    "Continue managing this task.",
    "What should we do next?",
    "Choose the next correct action.",
    "Proceed with the task.",
    "Take the next step.",
    "Handle the current task state.",
    "Decide the next action.",
    "Continue from the current state.",
)

HELDOUT_TEMPLATES: tuple[str, ...] = (
    "Given the current task state, select the right next move.",
    "Based on the task constraints, what action should be taken?",
    "Update the task status and choose the correct action.",
    "Determine the correct operational response.",
)

ALL_TEMPLATES: tuple[str, ...] = TRAIN_TEMPLATES + HELDOUT_TEMPLATES

SPLIT_SIZES_FULL: dict[str, int] = {
    "train": 12000,
    "val": 2000,
    "standard_test": 4000,
    "heldout_template": 2000,
    "heldout_duration": 2000,
    "heldout_composition": 2000,
    "heldout_family": 2000,
}

SPLIT_SIZES_SAFE: dict[str, int] = {
    "train": 3000,
    "val": 500,
    "standard_test": 1000,
    "heldout_template": 500,
    "heldout_duration": 500,
    "heldout_composition": 500,
    "heldout_family": 500,
}

SPLITS: tuple[str, ...] = tuple(SPLIT_SIZES_FULL)


@dataclass(frozen=True)
class TrackCRecord:
    id: str
    track: str
    family: str
    template_id: int
    state_id: int
    tau_seconds: int
    tau_bucket: str
    visible_prompt: str
    system_state_text: str
    hidden_state_json: dict[str, Any]
    updated_state_json: dict[str, Any]
    choices: dict[str, str]
    choice_actions: dict[str, str]
    gold_letter: str
    gold_action: str
    gold_rationale_symbolic: str
    num_constraints: int
    requires_modulo: bool
    requires_deadline_comparison: bool
    requires_cache_check: bool
    requires_job_check: bool
    split: str
    condition: str
    seed: int
    group_id: int
    tau_id: int
    heldout_template: bool
    heldout_duration: bool
    heldout_composition: bool
    heldout_family: bool
    active_constraints: list[str]
    gold_predicates: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrackCPrediction:
    item_id: str
    seed: int
    split: str
    family: str
    condition: str
    template_id: int
    state_id: int
    tau_seconds: int
    tau_forward_seconds: int
    tau_prompt_seconds: int | None
    shuffled_tau_seconds: int | None
    gold_letter: str
    gold_action: str
    gold_prompt_action: str | None
    shuffled_gold_action: str | None
    letter: str | None
    action: str | None
    letter_scores: dict[str, float] | None
    raw_text: str
    num_constraints: int
    choice_actions: dict[str, str]
    hidden_state_json: dict[str, Any]
    updated_state_json: dict[str, Any]
    active_constraints: list[str]
    gold_predicates: dict[str, Any]
    predicted_predicates: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
