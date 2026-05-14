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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--out", type=str, default="data/qwen_memrecall/train.jsonl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-distractors", type=int, default=4)
    p.add_argument("--max-distractors", type=int, default=14)
    args = p.parse_args()

    rng = random.Random(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(args.out, "w") as f:
        for i in range(args.n):
            nd = rng.randint(args.min_distractors, args.max_distractors)
            text, answer = gen_conversation(rng, nd)
            f.write(json.dumps({"text": text, "answer": answer}) + "\n")
            n_written += 1
    print(f"wrote {n_written} conversations -> {args.out}")


if __name__ == "__main__":
    main()
