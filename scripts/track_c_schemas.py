"""Compatibility import surface for Track C schemas."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.track_c.schemas import *  # noqa: F401,F403
