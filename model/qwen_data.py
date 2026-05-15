"""Generate synthetic memorize-recall conversations for Qwen IPCN training.

Each example is a multi-turn conversation:
  - The user states a FACT in turn 1 ("my favorite color is azure").
  - A variable number of DISTRACTOR turns follow (small-talk, unrelated
    questions, weather, etc).
  - The user then asks a RECALL question about the fact ("what's my
    favorite color?").
  - The assistant's answer should match the fact.

The point of this design is that the FACT is OUT OF CONTEXT by the time
the recall question arrives -- the conversation is longer than the
chunk_length, so without the memory bank, the assistant has no way to
know the answer. This makes memory dependence MEASURABLE (slot ablation
should hurt recall accuracy).

We generate in plain text, then tokenize with Qwen's tokenizer at load
time. Saved as JSONL with one conversation per line.

Usage:
  uv run python -m model.qwen_data --n 5000 --out data/qwen_memrecall/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


FACT_TEMPLATES: list[tuple[str, str, str]] = [
    # (statement, question, answer)
    ("My favorite color is {x}.", "What is my favorite color?", "{x}"),
    ("I was born on {x}.", "When was I born?", "{x}"),
    ("My password is {x}.", "What is my password?", "{x}"),
    ("My dog's name is {x}.", "What is my dog's name?", "{x}"),
    ("I work at {x}.", "Where do I work?", "{x}"),
    ("I live in {x}.", "Where do I live?", "{x}"),
    ("My phone number is {x}.", "What is my phone number?", "{x}"),
    ("My favorite food is {x}.", "What is my favorite food?", "{x}"),
    ("My car is a {x}.", "What kind of car do I have?", "{x}"),
    ("My best friend is {x}.", "Who is my best friend?", "{x}"),
    ("I graduated from {x}.", "Where did I graduate from?", "{x}"),
    ("My favorite book is {x}.", "What is my favorite book?", "{x}"),
    ("My favorite movie is {x}.", "What is my favorite movie?", "{x}"),
    ("My middle name is {x}.", "What is my middle name?", "{x}"),
    ("My lucky number is {x}.", "What is my lucky number?", "{x}"),
]

VALUE_POOLS: dict[str, list[str]] = {
    "color": ["azure", "vermillion", "chartreuse", "indigo", "magenta", "ochre",
              "periwinkle", "saffron", "teal", "umber", "amber", "cerulean"],
    "date": ["April 15", "October 3", "December 22", "June 8", "September 11",
             "February 29", "November 17", "January 5", "August 23", "March 30"],
    "password": ["sunset42", "blue9horse", "ironclad7", "moonfall88", "rainpath3",
                 "windgate15", "starflame91", "icebridge2", "treenest77", "skywatch6"],
    "name": ["Biscuit", "Pepper", "Apollo", "Luna", "Mango", "Ziggy", "Pumpkin",
             "Waffles", "Otter", "Pixel", "Banjo", "Maple"],
    "company": ["Atlas Robotics", "Vermillion Labs", "Polaris AI", "Heliconia Systems",
                "Lighthouse Analytics", "Sequoia Forge", "Indigo Networks", "Tide Foundry"],
    "city": ["Reykjavik", "Marrakesh", "Wellington", "Antwerp", "Tbilisi",
             "Bratislava", "Hobart", "Asheville", "Galway", "Ulaanbaatar"],
    "phone": ["555-0148", "555-2299", "555-7611", "555-3055", "555-8842", "555-1907"],
    "food": ["paella", "tagine", "okonomiyaki", "borscht", "khachapuri", "ramen",
             "ceviche", "bibimbap", "kishke", "satay"],
    "car": ["1992 Volvo 240", "2007 Subaru Legacy", "1968 Citroen DS", "2015 Honda Fit",
            "1983 Toyota Tercel"],
    "person": ["Tomas", "Lena", "Imani", "Yusuf", "Anya", "Diego", "Sasha"],
    "school": ["McGill", "Imperial College London", "ETH Zurich", "Reed College", "Caltech"],
    "book": ["A Wizard of Earthsea", "Stoner", "The Master and Margarita", "Beloved", "Pale Fire"],
    "movie": ["Stalker", "The Double Life of Veronique", "Russian Ark", "In the Mood for Love"],
    "middlename": ["Marlowe", "Theodora", "Iskander", "Verity", "Caspian", "Saoirse"],
    "number": ["8", "23", "42", "108", "1453", "1729", "31415", "7"],
}

# Map template index to which pool to draw from.
TEMPLATE_POOL_MAP = [
    "color", "date", "password", "name", "company", "city", "phone", "food",
    "car", "person", "school", "book", "movie", "middlename", "number",
]


DISTRACTORS = [
    "What's the weather like today?",
    "Can you tell me a joke?",
    "What's 47 times 13?",
    "Recommend a podcast for road trips.",
    "How do I make sourdough starter?",
    "Why is the sky blue?",
    "Tell me about black holes.",
    "What's a good first programming language?",
    "Translate hello to Japanese.",
    "What's the capital of Mongolia?",
    "Suggest a film for a rainy Sunday.",
    "How long does it take to learn piano?",
    "What's a haiku?",
    "Explain photosynthesis briefly.",
    "Name three rivers in Africa.",
    "What's tikka masala?",
    "Why do cats purr?",
    "Recommend an exercise for back pain.",
    "What's the speed of light?",
    "Tell me about the Renaissance.",
]


DISTRACTOR_REPLIES = [
    "I'd rather not get into that right now.",
    "Sorry, let's focus on something else.",
    "Hmm, I'm not the best resource for that.",
    "Maybe ask me later.",
    "That's a fun question for another time.",
    "Let's come back to that.",
    "I'll skip that one for now.",
]


def gen_conversation(rng: random.Random, n_distractors: int) -> tuple[str, str]:
    """Build one conversation as a single string (Qwen chat format) plus
    the canonical answer token string for eval.

    Returns: (conversation_text, expected_answer)
    """
    template_idx = rng.randrange(len(FACT_TEMPLATES))
    statement, question, answer_tpl = FACT_TEMPLATES[template_idx]
    pool = VALUE_POOLS[TEMPLATE_POOL_MAP[template_idx]]
    x = rng.choice(pool)
    statement = statement.format(x=x)
    answer = answer_tpl.format(x=x)

    turns: list[tuple[str, str]] = []
    turns.append(("user", statement))
    turns.append(("assistant", "Got it, I will remember that."))
    for _ in range(n_distractors):
        d = rng.choice(DISTRACTORS)
        r = rng.choice(DISTRACTOR_REPLIES)
        turns.append(("user", d))
        turns.append(("assistant", r))
    turns.append(("user", question))
    turns.append(("assistant", answer + "."))

    # Render in Qwen ChatML
    parts = []
    for role, content in turns:
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    text = "\n".join(parts)
    return text, answer


def gen_conversation_with_answer_span(
    rng: random.Random, n_distractors: int,
) -> tuple[str, str, str, str]:
    """Same as gen_conversation but also returns the recall question and
    the answer-only segment so the trainer can mask non-answer tokens.

    Returns: (full_text, answer, prefix_text, answer_text)
      full_text = prefix_text + answer_text
      prefix_text ends right at "<|im_start|>assistant\n" of the FINAL turn
      answer_text = "<answer>.<|im_end|>"

    The trainer feeds full_text, computes loss only on tokens whose
    position lies in answer_text. That way the LM loss focuses on
    learning to PRODUCE the answer rather than mimicking distractors.
    """
    template_idx = rng.randrange(len(FACT_TEMPLATES))
    statement, question, answer_tpl = FACT_TEMPLATES[template_idx]
    pool = VALUE_POOLS[TEMPLATE_POOL_MAP[template_idx]]
    x = rng.choice(pool)
    statement = statement.format(x=x)
    answer = answer_tpl.format(x=x)

    turns: list[tuple[str, str]] = []
    turns.append(("user", statement))
    turns.append(("assistant", "Got it, I will remember that."))
    for _ in range(n_distractors):
        d = rng.choice(DISTRACTORS)
        r = rng.choice(DISTRACTOR_REPLIES)
        turns.append(("user", d))
        turns.append(("assistant", r))
    turns.append(("user", question))
    # The final assistant turn is split for masking.
    prefix_parts = []
    for role, content in turns:
        prefix_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    prefix_text = "\n".join(prefix_parts) + "\n<|im_start|>assistant\n"
    answer_text = answer + ".<|im_end|>"
    full_text = prefix_text + answer_text
    return full_text, answer, prefix_text, answer_text


def _nonce_lower(rng: random.Random, k: int = 6) -> str:
    import string
    return "".join(rng.choices(string.ascii_lowercase, k=k))


def _nonce_cap(rng: random.Random, k: int = 6) -> str:
    return _nonce_lower(rng, k).capitalize()


NONCE_GEN = {
    # Maps the same TEMPLATE_POOL_MAP keys to nonce generators. Result is a
    # value drawn from ~26^k random strings rather than a pool of 10-12.
    "color":      lambda r: _nonce_lower(r, 7),
    "date":       lambda r: f"{r.choice(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])} {r.randint(1, 28)}",
    "password":   lambda r: _nonce_lower(r, 5) + str(r.randint(10, 99)),
    "name":       lambda r: _nonce_cap(r, 6),
    "company":    lambda r: _nonce_cap(r, 5) + " " + _nonce_cap(r, 5),
    "city":       lambda r: _nonce_cap(r, 7),
    "phone":      lambda r: f"555-{r.randint(0, 9999):04d}",
    "food":       lambda r: _nonce_lower(r, 7),
    "car":        lambda r: f"{r.randint(1980, 2024)} " + _nonce_cap(r, 6),
    "person":     lambda r: _nonce_cap(r, 5),
    "school":     lambda r: _nonce_cap(r, 6) + " University",
    "book":       lambda r: "The " + _nonce_cap(r, 6),
    "movie":      lambda r: _nonce_cap(r, 7),
    "middlename": lambda r: _nonce_cap(r, 6),
    "number":     lambda r: str(r.randint(1000, 99999)),
}


def gen_nonce_conversation(rng: random.Random, n_distractors: int) -> tuple[str, str, str, str]:
    """Same chat structure as gen_conversation_with_answer_span but the
    fact value is drawn from a HUGE nonce space rather than a 10-12
    element pool. Defeats template-guessing.

    Returns: (full_text, answer, prefix_text, answer_text)
    """
    template_idx = rng.randrange(len(FACT_TEMPLATES))
    statement, question, answer_tpl = FACT_TEMPLATES[template_idx]
    pool_key = TEMPLATE_POOL_MAP[template_idx]
    x = NONCE_GEN[pool_key](rng)
    statement = statement.format(x=x)
    answer = answer_tpl.format(x=x)

    turns: list[tuple[str, str]] = []
    # Randomize fact position so the model can't lock onto "turn 1 = fact".
    fact_position = rng.randint(0, max(0, n_distractors // 2))
    inserted_fact = False
    for j in range(n_distractors + 1):
        if j == fact_position:
            turns.append(("user", statement))
            ack_options = [
                "Got it, I will remember that.",
                "Noted.",
                "OK, I have that.",
                "Alright, I will keep that in mind.",
                "Understood.",
            ]
            turns.append(("assistant", rng.choice(ack_options)))
            inserted_fact = True
        else:
            d = rng.choice(DISTRACTORS)
            r = rng.choice(DISTRACTOR_REPLIES)
            turns.append(("user", d))
            turns.append(("assistant", r))
    if not inserted_fact:
        # Safety fallback: ensure fact is inserted somewhere.
        turns.append(("user", statement))
        turns.append(("assistant", "Got it, I will remember that."))
    turns.append(("user", question))

    prefix_parts = []
    for role, content in turns:
        prefix_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    prefix_text = "\n".join(prefix_parts) + "\n<|im_start|>assistant\n"
    answer_text = answer + ".<|im_end|>"
    full_text = prefix_text + answer_text
    return full_text, answer, prefix_text, answer_text


def gen_negative_conversation(rng: random.Random, n_distractors: int) -> tuple[str, str, str, str]:
    """No fact stated, recall question asked. Correct answer is 'I do not
    know.' Teaches the model that ABSENCE of memory means refuse, not
    confabulate a pool value.
    """
    template_idx = rng.randrange(len(FACT_TEMPLATES))
    _, question, _ = FACT_TEMPLATES[template_idx]
    turns: list[tuple[str, str]] = []
    for _ in range(n_distractors):
        d = rng.choice(DISTRACTORS)
        r = rng.choice(DISTRACTOR_REPLIES)
        turns.append(("user", d))
        turns.append(("assistant", r))
    turns.append(("user", question))
    prefix_parts = []
    for role, content in turns:
        prefix_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    prefix_text = "\n".join(prefix_parts) + "\n<|im_start|>assistant\n"
    answer_text = "I do not know.<|im_end|>"
    full_text = prefix_text + answer_text
    return full_text, "I do not know", prefix_text, answer_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--out", type=str, default="data/qwen_memrecall/train.jsonl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-distractors", type=int, default=4)
    p.add_argument("--max-distractors", type=int, default=14)
    p.add_argument("--with-answer-span", action="store_true")
    p.add_argument("--nonce", action="store_true",
                   help="use nonce string values (defeats template-guessing).")
    p.add_argument("--negative-frac", type=float, default=0.0,
                   help="fraction of examples that are negative controls "
                        "(no fact stated, recall -> 'I do not know').")
    args = p.parse_args()

    rng = random.Random(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_neg = 0
    with open(args.out, "w") as f:
        for i in range(args.n):
            nd = rng.randint(args.min_distractors, args.max_distractors)
            is_negative = rng.random() < args.negative_frac
            if is_negative:
                full_text, answer, prefix_text, answer_text = \
                    gen_negative_conversation(rng, nd)
                rec = {
                    "text": full_text, "answer": answer,
                    "prefix_text": prefix_text, "answer_text": answer_text,
                    "mode": "negative",
                }
                n_neg += 1
            elif args.nonce:
                full_text, answer, prefix_text, answer_text = \
                    gen_nonce_conversation(rng, nd)
                rec = {
                    "text": full_text, "answer": answer,
                    "prefix_text": prefix_text, "answer_text": answer_text,
                    "mode": "nonce",
                }
            elif args.with_answer_span:
                full_text, answer, prefix_text, answer_text = \
                    gen_conversation_with_answer_span(rng, nd)
                rec = {
                    "text": full_text, "answer": answer,
                    "prefix_text": prefix_text, "answer_text": answer_text,
                    "mode": "pool",
                }
            else:
                text, answer = gen_conversation(rng, nd)
                rec = {"text": text, "answer": answer, "mode": "pool"}
            f.write(json.dumps(rec) + "\n")
            n_written += 1
    print(f"wrote {n_written} conversations -> {args.out} (negative: {n_neg})")


if __name__ == "__main__":
    main()
