"""Safety watchdog. Runs alongside training, kills it if hard failure.

Checks every N seconds:
  - Training process still alive
  - LM loss not NaN
  - LM loss not 10x worse than rolling mean
  - Disk space sufficient

If any condition fires, sends SIGTERM to the training PID and writes a
report.

Usage:
  # In tmux pane 1: training
  nohup uv run python -m model.run_phase --phase 0 --steps 100000 ... &
  # In tmux pane 2: safety
  uv run python -m scripts.safety --pid $! --log logs/phase0_sanity.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import sys
import time
from collections import deque
from pathlib import Path


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_training(pid: int, reason: str, alert_path: str):
    Path(alert_path).parent.mkdir(parents=True, exist_ok=True)
    with open(alert_path, "a") as f:
        f.write(json.dumps({"time": time.time(), "kind": "training_killed",
                            "pid": pid, "reason": reason}) + "\n")
    print(f"!! KILLING training pid={pid}: {reason}")
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(5)
        if is_alive(pid):
            os.kill(pid, signal.SIGKILL)
    except OSError as e:
        print(f"   already gone: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pid", type=int, required=True, help="training process PID")
    p.add_argument("--log", type=str, required=True, help="training log JSONL path")
    p.add_argument("--check-every", type=float, default=30.0, help="seconds between checks")
    p.add_argument("--lm-explode-factor", type=float, default=10.0,
                   help="kill if LM > factor * rolling mean")
    p.add_argument("--min-free-gb", type=float, default=10.0,
                   help="kill if free disk drops below this")
    p.add_argument("--alert-path", type=str, default="reports/alerts.jsonl")
    args = p.parse_args()

    log_path = Path(args.log)
    print(f"Safety watchdog: monitoring pid={args.pid}, log={log_path}, "
          f"check every {args.check_every}s")

    lm_window: deque[float] = deque(maxlen=50)
    last_pos = 0

    while True:
        if not is_alive(args.pid):
            print("Training process ended normally.")
            sys.exit(0)

        # Read new log lines
        if log_path.exists():
            with open(log_path, "r") as f:
                f.seek(last_pos)
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:                          # noqa: BLE001
                        continue
                    if "lm_loss" not in rec:
                        continue
                    lm = rec["lm_loss"]
                    if math.isnan(lm) or math.isinf(lm):
                        kill_training(args.pid, f"NaN/Inf LM loss at step {rec.get('step')}",
                                      args.alert_path)
                        sys.exit(2)
                    if len(lm_window) >= 10:
                        mean = sum(lm_window) / len(lm_window)
                        if lm > mean * args.lm_explode_factor:
                            kill_training(
                                args.pid,
                                f"LM loss exploded: {lm:.2f} > {args.lm_explode_factor}x "
                                f"trailing mean {mean:.2f} at step {rec.get('step')}",
                                args.alert_path,
                            )
                            sys.exit(2)
                    lm_window.append(lm)
                last_pos = f.tell()

        # Disk space check
        free_gb = shutil.disk_usage(str(log_path.parent.parent)).free / 1e9
        if free_gb < args.min_free_gb:
            kill_training(args.pid, f"disk space critical: {free_gb:.1f} GB free",
                          args.alert_path)
            sys.exit(2)

        time.sleep(args.check_every)


if __name__ == "__main__":
    main()
