"""Compatibility entry point for the Track C dataset generator."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.track_c.generate import main


if __name__ == "__main__":
    raise SystemExit(main())
