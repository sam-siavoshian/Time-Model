"""Compatibility entry point for Track C diagnostic baselines."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.track_c.baselines import main


if __name__ == "__main__":
    raise SystemExit(main())
