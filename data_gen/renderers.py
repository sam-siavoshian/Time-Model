"""Canonical text renderers for each dataset type.

Each renderer takes a single example dict (loaded from JSONL) and returns
the text string that the model sees during training. Used by tokenize_all.py
to build binary caches.

Conventions:
- Use ASCII-only delimiters when possible (works in any tokenizer).
- Mark phase boundaries clearly so probes can extract per-phase activations.
- Keep timestamps in the visible text (the model sees them; chronometric
  encoding χ_t is delivered separately via the architecture).
"""

from __future__ import annotations

from typing import Any


# ---------- Latent World ----------

def render_latent_world(stream: dict[str, Any]) -> str:
    lines: list[str] = ["<|stream|>"]
    for ev in stream["events"]:
        lines.append(ev["text"])
    lines.append("<|questions|>")
    for q in stream["questions"]:
        lines.append(f"Q: {q['text']}")
        lines.append(f"A: {q['answer']}")
    lines.append("<|endofstream|>")
    return "\n".join(lines)


# ---------- Memory-Biased Ambiguity ----------

def render_ambiguity(ex: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("<|memory|>")
    for fact in ex["phase1_memory"]:
        lines.append(f"- {fact}")
    lines.append("<|input|>")
    lines.append(ex["phase2_input"])
    lines.append("<|question|>")
    lines.append(ex["phase3_question"])
    for key, opt in ex["options"].items():
        lines.append(f"{key}. {opt}")
    lines.append(f"<|answer|>{ex['correct_answer']}<|endofexample|>")
    return "\n".join(lines)


# ---------- Consolidation Ladder ----------

def render_consolidation(rule: dict[str, Any]) -> str:
    lines: list[str] = ["<|intro|>", rule["intro_chunk"], "<|queries|>"]
    for q in rule["query_chunks"]:
        lines.append(q)
    lines.append("<|removal_test|>")
    lines.append(rule["removal_test_chunk"])
    lines.append("<|contradiction|>")
    lines.append(rule["contradiction_chunk"])
    lines.append("<|endofrule|>")
    return "\n".join(lines)


# ---------- Chronometric Pair ----------

def render_chronometric_pair(pair: dict[str, Any], arm: str) -> str:
    """arm in {'real', 'ablated'} — selects which Δτ to embed in the rendered text.

    Note: in actual model training, the chronometric vector χ_t is delivered
    via architecture, not via this string. This rendering is for cache
    storage + sanity dumps.
    """
    delta = (
        pair["delta_tau_real_minutes"] if arm == "real"
        else pair["delta_tau_ablated_minutes"]
    )
    lines: list[str] = [
        f"<|delta_tau_minutes|>{delta}",
        "<|visible|>",
        pair["visible_text"],
        "<|duration_sensitive_q|>",
        pair["duration_sensitive_q"]["text"],
        f"A: {pair['duration_sensitive_q']['answer']}",
        "<|duration_insensitive_q|>",
        pair["duration_insensitive_q"]["text"],
        f"A: {pair['duration_insensitive_q']['answer']}",
        "<|endofpair|>",
    ]
    return "\n".join(lines)


# ---------- Contradiction Pair ----------

def render_contradiction_pair(pair: dict[str, Any], arm: str) -> str:
    """arm in {'mem_a_amb', 'mem_b_amb', 'mem_a_exp', 'mem_b_exp'}."""
    use_a = arm.startswith("mem_a")
    use_explicit = arm.endswith("exp")
    facts = pair["memory_a_facts"] if use_a else pair["memory_b_facts"]
    inp = pair["explicit_input"] if use_explicit else pair["ambiguous_input"]
    if use_explicit:
        correct = pair["correct_under_explicit"]
    else:
        correct = pair["correct_under_memory_a"] if use_a else pair["correct_under_memory_b"]

    lines: list[str] = ["<|memory|>"]
    for f in facts:
        lines.append(f"- {f}")
    lines.append("<|input|>")
    lines.append(inp)
    lines.append("<|question|>")
    lines.append(pair["question"])
    for key, opt in pair["options"].items():
        lines.append(f"{key}. {opt}")
    lines.append(f"<|answer|>{correct}<|endofexample|>")
    return "\n".join(lines)


# ---------- Real text ----------

def render_real_text(chunk: dict[str, Any]) -> str:
    return chunk["text"]


# ---------- Dispatch ----------

RENDERERS = {
    "latent_world":        render_latent_world,
    "ambiguity":           render_ambiguity,
    "consolidation":       render_consolidation,
    "chronometric_pair":   render_chronometric_pair,
    "contradiction_pair":  render_contradiction_pair,
    "real_text":           render_real_text,
}
