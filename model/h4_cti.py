"""H4 Consolidation Transfer Index (CTI).

CTI measures whether usage-driven consolidation actually migrates a slot's
behavioral effect into LoRA weights.

For a slot m_i used heavily for a rule:
  Acc_pre^with     = accuracy WITH slot, before consolidation
  Acc_pre^without  = accuracy WITHOUT slot, before consolidation
  Acc_post^with    = accuracy WITH slot, after consolidation
  Acc_post^without = accuracy WITHOUT slot, after consolidation

  CTI_i = (Acc_post^without - Acc_pre^without)
          / (Acc_pre^with - Acc_pre^without + eps)

If consolidation worked, removing the slot AFTER consolidation should NOT
hurt nearly as much, because the LoRA captured the behavior. CTI > 0.7 means
70%+ of the slot's effect was successfully transferred to weights.

Test substrate: Consolidation Ladder dataset
(data/consolidation/ladder_train.jsonl). Each rule has intro + query chunks
+ removal_test. For evaluation:
  - Run model on a queries-only test split with the slot active vs removed
  - Pre-consolidation = checkpoint BEFORE consolidation pass
  - Post-consolidation = checkpoint AFTER consolidation pass
  - Both with/without slot via memory zero-out

Requires two checkpoints: pre and post. For prototype/Phase 0 we can pass
the SAME checkpoint twice (CTI will be 0 since nothing was consolidated).
After Phase 1 training, pre = phase0 checkpoint, post = phase1 checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import tiktoken
import torch
import torch.nn.functional as F

from model.checkpoint import load_checkpoint
from model.config import IPCNConfig
from model.h7_contradiction import render_memory_block
from model.ipcn import IPCN


@torch.no_grad()
def _accuracy_on_queries(
    model: IPCN,
    enc,
    intro_chunk: str,
    query_chunks: list[str],
    answers: list[str],
    chunk_length: int,
    slot_to_zero: int | None = None,
) -> float:
    """Reset, feed intro, run queries, compare answer tokens."""
    model.reset_memory()
    intro_toks = torch.tensor(enc.encode(intro_chunk), dtype=torch.long)
    if intro_toks.numel() < chunk_length:
        intro_toks = F.pad(intro_toks, (0, chunk_length - intro_toks.numel()), value=0)
    else:
        intro_toks = intro_toks[:chunk_length]
    intro_out = model.forward_chunk(intro_toks, tau_t=0.0, delta_tau=0.0)
    L = intro_toks.shape[0]
    h_content = intro_out.hidden_last[model.cfg.prefix_length:]
    novelty = 1.0 - F.normalize(model.memory.W_k_m(h_content), dim=-1) @ F.normalize(model.memory.k, dim=-1).t()
    novelty = novelty.max(dim=-1).values * -1 + 1.0
    chi_t = model.chrono(tau=torch.tensor([0.0]), delta_tau=torch.tensor([0.0])).squeeze(0)
    model.memory.write(
        h_L=h_content, b=intro_out.b_broadcast, z=model.z,
        surprise=torch.zeros(L), novelty=novelty, u_prefix=torch.zeros(L),
        tau_t=0.0, chi_t=chi_t,
    )

    # Optionally zero a slot to test the "without slot" arm
    if slot_to_zero is not None:
        with torch.no_grad():
            model.memory.k[slot_to_zero].zero_()
            model.memory.v[slot_to_zero].zero_()

    correct = 0
    total = 0
    for q_text, answer in zip(query_chunks, answers):
        if not answer:
            continue
        q_toks = torch.tensor(enc.encode(q_text), dtype=torch.long)
        if q_toks.numel() < chunk_length:
            q_toks = F.pad(q_toks, (0, chunk_length - q_toks.numel()), value=0)
        else:
            q_toks = q_toks[:chunk_length]
        q_out = model.forward_chunk(q_toks, tau_t=1.0, delta_tau=1.0)
        # The answer token IDs we want to predict are the first 1-2 tokens
        # of `answer`. We score whether the predicted next-token equals the
        # first answer token at the position where the query text ends.
        # Heuristic: take the last logits row that isn't padding.
        valid_len = (q_toks != 0).sum().item()
        pred = q_out.logits[valid_len - 1].argmax().item()
        ans_toks = enc.encode(answer.strip())
        if ans_toks and pred == ans_toks[0]:
            correct += 1
        total += 1
    return correct / max(1, total)


def _extract_query_answers(query_chunk: str) -> tuple[str, str]:
    """A query chunk like 'Mara is a engineer. Are they permitted...? Answer: yes.'
    Split on 'Answer:' so the model is asked to predict the answer token."""
    if "Answer:" in query_chunk:
        prompt, ans = query_chunk.split("Answer:", 1)
        return prompt + "Answer:", ans.strip().rstrip(".")
    return query_chunk, ""


def compute_CTI(
    pre_model: IPCN,
    post_model: IPCN,
    rules_subset: list[dict],
    chunk_length: int,
    eps: float = 1e-6,
) -> dict:
    """Compute CTI averaged across a subset of rules. For each rule:

    1. Feed intro_chunk to write the rule slot.
    2. Identify slot with highest usage (the "owner" of this rule).
    3. Score accuracy on first K=5 query chunks with slot active vs zeroed.
    4. Repeat under both pre and post models.
    5. CTI_i = (Acc_post_without - Acc_pre_without) / (Acc_pre_with - Acc_pre_without + eps)
    """
    enc = tiktoken.get_encoding("gpt2")
    cti_values = []
    pre_with_all, pre_without_all = [], []
    post_with_all, post_without_all = [], []

    for rule in rules_subset:
        prompts_answers = [_extract_query_answers(q) for q in rule["query_chunks"][:5]]
        prompts = [p for p, _ in prompts_answers]
        answers = [a for _, a in prompts_answers]
        if not any(answers):
            continue

        # PRE: identify primary slot by writing intro then reading usage
        pre_model.reset_memory()
        intro_toks = torch.tensor(enc.encode(rule["intro_chunk"]), dtype=torch.long)
        if intro_toks.numel() < chunk_length:
            intro_toks = F.pad(intro_toks, (0, chunk_length - intro_toks.numel()), value=0)
        else:
            intro_toks = intro_toks[:chunk_length]
        intro_out = pre_model.forward_chunk(intro_toks, tau_t=0.0, delta_tau=0.0)
        L = intro_toks.shape[0]
        h_content = intro_out.hidden_last[pre_model.cfg.prefix_length:]
        novelty = 1.0 - F.normalize(pre_model.memory.W_k_m(h_content), dim=-1) @ F.normalize(pre_model.memory.k, dim=-1).t()
        novelty = novelty.max(dim=-1).values * -1 + 1.0
        chi_t = pre_model.chrono(tau=torch.tensor([0.0]), delta_tau=torch.tensor([0.0])).squeeze(0)
        pre_model.memory.write(
            h_L=h_content, b=intro_out.b_broadcast, z=pre_model.z,
            surprise=torch.zeros(L), novelty=novelty, u_prefix=torch.zeros(L),
            tau_t=0.0, chi_t=chi_t,
        )
        # Primary slot: highest tau_write (most recently written this turn) OR
        # we can pick the slot with max key norm. Use max conflict (most recently touched).
        primary_slot = pre_model.memory.tau_write.argmax().item()

        acc_pre_with = _accuracy_on_queries(pre_model, enc, rule["intro_chunk"], prompts, answers, chunk_length, slot_to_zero=None)
        acc_pre_without = _accuracy_on_queries(pre_model, enc, rule["intro_chunk"], prompts, answers, chunk_length, slot_to_zero=primary_slot)
        acc_post_with = _accuracy_on_queries(post_model, enc, rule["intro_chunk"], prompts, answers, chunk_length, slot_to_zero=None)
        acc_post_without = _accuracy_on_queries(post_model, enc, rule["intro_chunk"], prompts, answers, chunk_length, slot_to_zero=primary_slot)

        cti = (acc_post_without - acc_pre_without) / (acc_pre_with - acc_pre_without + eps)
        cti_values.append(cti)
        pre_with_all.append(acc_pre_with)
        pre_without_all.append(acc_pre_without)
        post_with_all.append(acc_post_with)
        post_without_all.append(acc_post_without)

    return {
        "n_rules": len(cti_values),
        "CTI_mean": mean(cti_values) if cti_values else 0.0,
        "Acc_pre_with":     mean(pre_with_all)     if pre_with_all else 0.0,
        "Acc_pre_without":  mean(pre_without_all)  if pre_without_all else 0.0,
        "Acc_post_with":    mean(post_with_all)    if post_with_all else 0.0,
        "Acc_post_without": mean(post_without_all) if post_without_all else 0.0,
        "threshold": 0.7,
        "passes": (mean(cti_values) if cti_values else 0.0) > 0.7,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rules", type=str, default="data/consolidation/ladder_train.jsonl")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--pre-checkpoint", type=str, default=None)
    p.add_argument("--post-checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    torch.manual_seed(0)
    if args.pre_checkpoint:
        pre_model, _, cfg, _ = load_checkpoint(args.pre_checkpoint, map_location=args.device)
    else:
        cfg = IPCNConfig()
        pre_model = IPCN(cfg).to(args.device)
    if args.post_checkpoint:
        post_model, _, _, _ = load_checkpoint(args.post_checkpoint, map_location=args.device)
    else:
        post_model = pre_model                                # CTI will be 0 (no consolidation happened)
    pre_model.train(False)
    post_model.train(False)

    rules = []
    with open(args.rules) as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            rules.append(json.loads(line))

    print(f"=== H4 CTI eval (n={len(rules)} rules) ===")
    r = compute_CTI(pre_model, post_model, rules, chunk_length=cfg.chunk_length)
    for k, v in r.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
