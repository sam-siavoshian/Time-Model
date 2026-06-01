"""Compatibility entry point for Track C CI forced-choice training."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.qwen_time_train import main


if __name__ == "__main__":
    if not any(arg == "--loss-mode" or arg.startswith("--loss-mode=") for arg in sys.argv[1:]):
        sys.argv.extend(["--loss-mode", "forced_choice"])
    raise SystemExit(main())
