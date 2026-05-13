"""Pydantic schemas for all IPCN training datasets.

One file owns the data contracts. Every generator + loader imports from here.
Designed for HuggingFace release: stable field names, plain JSON-serializable,
includes provenance fields (seed, version) for reproducibility.
"""

from __future__ import annotations

from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.1.0"


# ---------- Temporal Latent World ----------

class WorldEvent(BaseModel):
    """One event in a Temporal Latent World stream."""

    tau_minutes: float                                       # real elapsed time within stream
    delta_tau_minutes: float                                 # gap since previous event
    text: str                                                # human-readable line
    event_type: Literal[
        "state_change", "rule_intro", "silent_gap", "question"
    ]
    affected_entities: list[str] = Field(default_factory=list)
    hidden_state_snapshot: Optional[dict] = None             # full world state after event


class LatentWorldQuestion(BaseModel):
    """A question about hidden world state."""

    text: str
    answer: str
    rationale: str                                           # how answer follows from events + elapsed time
    duration_sensitive: bool                                 # changes when Δτ changes?
    question_type: Literal[
        "current_state", "predict_future", "explain_past",
        "decay", "delayed_transition", "periodic_phase",
    ]


class LatentWorldStream(BaseModel):
    """One Temporal Latent World training sample."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    seed: int                                                # for reproducibility
    schema_version: str = SCHEMA_VERSION
    duration_minutes: float
    events: list[WorldEvent]
    questions: list[LatentWorldQuestion]
    target_token_length: int                                 # nominal bin: 1k, 2k, ..., 128k


# ---------- Memory-Biased Ambiguity Suite ----------

class AmbiguityExample(BaseModel):
    """One ambiguity example. Three phases: memory, ambiguous input, question."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    schema_version: str = SCHEMA_VERSION
    family: str                                              # e.g. "object_sense_bat"
    memory_state_id: str                                     # e.g. "baseball_priors" — for swap tests
    phase1_memory: list[str]                                 # ordered facts (the prior)
    phase2_input: str                                        # ambiguous current input
    phase3_question: str
    options: dict[str, str]                                  # option_key -> text, e.g. {"A": "..."}
    correct_answer: str                                      # option key, e.g. "B"


# ---------- Consolidation Ladder ----------

class ConsolidationRule(BaseModel):
    """One rule for testing slot → weight migration."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    schema_version: str = SCHEMA_VERSION
    rule_id: str                                             # e.g. "rin_lies_about_blue"
    rule_text: str
    intro_chunk: str                                         # establishes the rule
    query_chunks: list[str]                                  # contexts using the rule
    removal_test_chunk: str                                  # context after slot is removed
    contradiction_chunk: str                                 # contradictory evidence


# ---------- Chronometric Ablation Pairs (Prediction 6 eval) ----------

class ChronometricPair(BaseModel):
    """Paired streams: same tokens, different Δτ. Tests Δτ as substrate."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    schema_version: str = SCHEMA_VERSION
    paired_stream_id: str                                    # links real-Δτ and ablated-Δτ versions
    delta_tau_real_minutes: float
    delta_tau_ablated_minutes: float                         # constant for ablation arm
    visible_text: str                                        # IDENTICAL across both versions
    duration_sensitive_q: LatentWorldQuestion
    duration_insensitive_q: LatentWorldQuestion


# ---------- Contradiction Pairs (Prediction 7 eval) ----------

class ContradictionPair(BaseModel):
    """Memory swap test under ambiguous vs explicit input."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    schema_version: str = SCHEMA_VERSION
    memory_state_a_id: str                                   # e.g. "baseball_priors"
    memory_state_b_id: str                                   # e.g. "cave_priors"
    memory_a_facts: list[str]
    memory_b_facts: list[str]
    ambiguous_input: str
    explicit_input: str                                      # resolves the ambiguity
    question: str
    options: dict[str, str]
    correct_under_memory_a: str
    correct_under_memory_b: str
    correct_under_explicit: str                              # should be fixed regardless of memory


# ---------- Real-text mix (just a thin wrapper) ----------

class RealTextChunk(BaseModel):
    """Slice of OpenWebText / C4 for Phase 3 mixing."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    source: Literal["openwebtext", "c4", "wikipedia", "other"]
    text: str
    token_count: Optional[int] = None
