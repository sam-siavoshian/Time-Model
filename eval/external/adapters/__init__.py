"""Adapter registry for the tau_sessions external benchmark.

An adapter is a thin object with a single contract: given a (prompt, tau)
pair, return the model's generated string. The benchmark harness scores
generations against per-task ground truth.

Three adapters ship in this repo:
  vanilla  -- frozen base LLM, no tau channel and no tau text. The
              "what does the base model know about elapsed time"
              baseline.
  prompt   -- frozen base LLM with tau injected as a textual prefix
              ("[elapsed: 3h 42m] ..."). The string-conditioning
              baseline.
  ci       -- Chronometric Injection: base LLM wrapped with the v15
              LoRA + per-layer FiLM chrono channel. tau is fed as a
              float tensor to the chrono encoder.

Third parties can add adapters by subclassing TauAdapter, implementing
load() and generate(), and registering the class in the ADAPTERS dict
below (or via the --adapter-path flag on the eval CLI).

License: MIT.
"""

from __future__ import annotations

from .base import TauAdapter

__all__ = ["TauAdapter", "load_adapter", "ADAPTERS"]


def load_adapter(name: str, **kwargs) -> TauAdapter:
    """Instantiate and load an adapter by short name."""
    if name not in ADAPTERS:
        raise KeyError(
            f"unknown adapter {name!r}. Known: {sorted(ADAPTERS)}. "
            f"Add a new adapter under eval/external/adapters/ and register it here."
        )
    cls = ADAPTERS[name]
    adapter = cls(**kwargs)
    adapter.load()
    return adapter


# Lazy registry. We import here (not at module top) so a missing optional
# dep (e.g. a torch problem during ci_adapter import) does not break the
# whole package import for the vanilla / prompt baselines.
def _build_registry() -> dict[str, type[TauAdapter]]:
    from .vanilla_adapter import VanillaAdapter
    from .prompt_adapter import PromptAdapter
    from .ci_adapter import CIAdapter
    return {
        "vanilla": VanillaAdapter,
        "prompt":  PromptAdapter,
        "ci":      CIAdapter,
    }


ADAPTERS = _build_registry()
