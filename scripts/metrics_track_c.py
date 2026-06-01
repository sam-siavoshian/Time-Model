"""Compatibility import surface for Track C metrics."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.track_c.metrics import (  # noqa: F401
    accuracy,
    balanced_accuracy,
    conflict_follow,
    mean_std,
    split_breakdowns,
    state_accuracy,
    wrong_state_consistency,
)
