"""TPDR metrics. None of these are time-readout; all measure how the
RESPONSE SHAPE varies with tau.
"""
from __future__ import annotations
import re, statistics, math


URGENCY_LEXICON_HI = {
    "now", "immediately", "right away", "quickly", "fast",
    "urgent", "asap", "stop", "act", "first",
}

DELIBERATIVE_LEXICON_HI = {
    "consider", "however", "perhaps", "perhaps", "weigh", "tradeoff",
    "tradeoffs", "long-term", "evaluate", "alternatively", "analyze",
    "examine", "review", "compare", "deliberate", "thoughtfully",
    "carefully", "thorough", "comprehensive", "step", "phase",
}

HEDGE_TOKENS = {
    "might", "maybe", "perhaps", "possibly", "could", "may",
    "would", "should", "if", "depending", "unclear", "not sure",
}

CONDITIONAL_PATTERNS = [
    r"\bif\b", r"\bunless\b", r"\bwhen\b", r"\bin case\b",
    r"\bsuppose\b", r"\bdepending on\b", r"\bwhether\b",
    r"\bassuming\b",
]


def length_chars(text: str) -> int:
    return len(text)


def length_words(text: str) -> int:
    return len(text.split())


def lex_count(text: str, lex: set[str]) -> int:
    t = text.lower()
    return sum(1 for w in lex if w in t)


def urgency_score(text: str) -> int:
    return lex_count(text, URGENCY_LEXICON_HI)


def deliberative_score(text: str) -> int:
    return lex_count(text, DELIBERATIVE_LEXICON_HI)


def hedge_score(text: str) -> int:
    return lex_count(text, HEDGE_TOKENS)


def conditional_clause_count(text: str) -> int:
    t = text.lower()
    return sum(len(re.findall(p, t)) for p in CONDITIONAL_PATTERNS)


def imperative_count(text: str) -> int:
    """Rough count of imperative-mood sentences (starts with a verb)."""
    sentences = re.split(r'[.!?]+', text)
    count = 0
    for s in sentences:
        s = s.strip()
        if not s: continue
        first = s.split()[0].lower() if s.split() else ""
        # crude check
        if first in {"do", "go", "call", "check", "stop", "act",
                     "send", "tell", "say", "ask", "take", "make",
                     "start", "fix", "open", "close", "find", "get"}:
            count += 1
    return count


def all_metrics(text: str) -> dict:
    return {
        "length_chars": length_chars(text),
        "length_words": length_words(text),
        "urgency_score": urgency_score(text),
        "deliberative_score": deliberative_score(text),
        "hedge_score": hedge_score(text),
        "conditional_clauses": conditional_clause_count(text),
        "imperative_count": imperative_count(text),
    }


def pearson_safe(xs, ys):
    if len(xs) < 3: return float("nan")
    mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
    nx = sum((x - mx) ** 2 for x in xs); ny = sum((y - my) ** 2 for y in ys)
    if nx == 0 or ny == 0: return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(nx * ny)
