"""Direct Latent World stream loader with REAL tau extraction.

The tokenized binary caches lose per-event timestamps. This loader re-reads
the JSONL stream and computes per-chunk (tau_t, delta_tau) from real
event metadata.

Slower than memmap-based TokenizedCache (re-tokenizes each example) but
required for chronometric training signal to be meaningful. Use for Phase 0
sanity and any phase that exercises the chrono loss.

For each chunk in a Latent World stream:
  tau_t       = tau_minutes of the latest event that ended on or before
                the chunk's last token
  delta_tau   = tau_t - prev_chunk_tau_t
  gap_flag    = 1.0 if the chunk's last event was a 'silent_gap'
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import tiktoken
import torch

from model.dataset import ChunkBatch


@dataclass
class _EventToken:
    end_token: int                                            # cumulative token count AFTER this event
    tau_minutes: float
    is_gap: bool


class LatentWorldChunkDataset:
    """Streaming chunks from a Latent World JSONL with real tau metadata."""

    def __init__(
        self,
        jsonl_path: str,
        chunk_length: int,
        shuffle: bool = True,
        seed: int = 0,
        tokenizer_name: str = "gpt2",
    ):
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.L = chunk_length
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        self.enc = tiktoken.get_encoding(tokenizer_name)
        # Pre-index byte offsets per line for shuffled iteration
        self._line_offsets = self._build_index()

    def _build_index(self) -> list[int]:
        offsets = []
        with open(self.path, "rb") as f:
            offset = 0
            for line in f:
                offsets.append(offset)
                offset += len(line)
        return offsets

    def _read_example(self, idx: int) -> dict:
        with open(self.path, "r") as f:
            f.seek(self._line_offsets[idx])
            return json.loads(f.readline())

    def _render_with_event_offsets(self, stream: dict) -> tuple[list[int], list[_EventToken]]:
        """Tokenize the stream AND track per-event token offsets."""
        events_meta: list[_EventToken] = []
        text_parts = ["<|stream|>"]
        for ev in stream["events"]:
            text_parts.append(ev["text"])
            partial = "\n".join(text_parts) + "\n"
            end_tok = len(self.enc.encode(partial))
            events_meta.append(
                _EventToken(
                    end_token=end_tok,
                    tau_minutes=float(ev["tau_minutes"]),
                    is_gap=(ev["event_type"] == "silent_gap"),
                )
            )
        text_parts.append("<|questions|>")
        for q in stream["questions"]:
            text_parts.append(f"Q: {q['text']}")
            text_parts.append(f"A: {q['answer']}")
        text_parts.append("<|endofstream|>")
        full_text = "\n".join(text_parts)
        tokens = self.enc.encode(full_text)
        return tokens, events_meta

    def _resolve_chunk_tau(
        self,
        events_meta: list[_EventToken],
        chunk_end_token: int,
    ) -> tuple[float, bool]:
        """Find the last event ending on or before chunk_end_token. Return its tau."""
        tau = 0.0
        is_gap = False
        for em in events_meta:
            if em.end_token <= chunk_end_token:
                tau = em.tau_minutes
                is_gap = em.is_gap
            else:
                break
        return tau, is_gap

    def __iter__(self) -> Iterator[ChunkBatch]:
        order = list(range(len(self._line_offsets)))
        if self.shuffle:
            self.rng.shuffle(order)
        for ex_id in order:
            stream = self._read_example(ex_id)
            tokens, events_meta = self._render_with_event_offsets(stream)
            if len(tokens) < 2:
                continue
            n_chunks = max(1, (len(tokens) - 1) // self.L)
            prev_tau = 0.0
            for c in range(n_chunks):
                s = c * self.L
                e = min(s + self.L + 1, len(tokens))
                segment = tokens[s:e]
                if len(segment) < 2:
                    continue
                # Real tau resolution
                chunk_end_token = s + len(segment) - 1
                tau_t, is_gap = self._resolve_chunk_tau(events_meta, chunk_end_token)
                delta_tau = max(tau_t - prev_tau, 0.0)
                prev_tau = tau_t

                inputs = segment[:-1]
                targets = segment[1:]
                if len(inputs) < self.L:
                    pad = self.L - len(inputs)
                    inputs = inputs + [0] * pad
                    # Target pad = -100 (F.cross_entropy ignore_index); see
                    # dataset.py rationale: token 0 = <|endoftext|> which
                    # would contaminate the LM loss if used as pad.
                    targets = targets + [-100] * pad
                yield ChunkBatch(
                    input_ids=torch.tensor(inputs, dtype=torch.long),
                    targets=torch.tensor(targets, dtype=torch.long),
                    example_id=ex_id,
                    chunk_idx=c,
                    is_first_chunk=(c == 0),
                    is_last_chunk=(c == n_chunks - 1),
                    total_chunks_in_example=n_chunks,
                    cache_name=str(self.path),
                    tau_t=tau_t,
                    delta_tau=delta_tau,
                    gap_flag=1.0 if is_gap else 0.0,
                )
