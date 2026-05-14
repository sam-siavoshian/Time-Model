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


def _has_completion_sentinel(log_path: Path) -> tuple[bool, dict | None]:
    """Scan the log for a `training_complete` event written by train_loop.

    Returns (found, record). When training exits cleanly, train_loop writes:
      {"event": "training_complete", "step": N, "max_steps": M,
       "reason": "max_steps" | "iterator_exhausted", "time": ...}
    Absence of this sentinel means the training process died without
    finishing the loop -- i.e. it crashed.

    Bare `except Exception` is intentional here: we want to detect
    `training_complete` no matter what other junk surrounds it in the
    log. Failures to parse one line should never abort the scan.
    """
    if not log_path.exists():
        return False, None
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
    except OSError:
        return False, None
    # Walk backward; completion sentinel is the last meaningful record.
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:                                     # noqa: BLE001
            continue
        if rec.get("event") == "training_complete":
            return True, rec
    return False, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pid", type=int, required=True, help="training process PID")
    p.add_argument("--log", type=str, required=True, help="training log JSONL path")
    p.add_argument("--check-every", type=float, default=30.0, help="seconds between checks")
    p.add_argument("--lm-explode-factor", type=float, default=10.0,
                   help="kill if LM > factor * rolling mean")
    p.add_argument(
        "--lm-explode-consecutive", type=int, default=3,
        help="number of CONSECUTIVE chunks above the explode factor "
             "before kill_training fires. Set >1 (default 3) so per-chunk "
             "loss variance on mixed datasets does not false-fire. Phase 1 "
             "data includes both easy LM chunks (loss ~0.5) and hard rule "
             "chunks (loss 15+ until memory is wired into behavior); a "
             "single 15+ reading is data noise, not divergence. Three "
             "consecutive 15+ readings IS divergence.",
    )
    p.add_argument("--min-free-gb", type=float, default=10.0,
                   help="kill if free disk drops below this")
    p.add_argument(
        "--stall-secs", type=float, default=600.0,
        help="kill if process is alive but log shows no new progress "
             "(no new lm_loss or consolidation records) within this many "
             "seconds. Catches dataloader deadlocks, NCCL collective hangs, "
             "and other live-but-stuck failure modes that NaN/explosion "
             "checks cannot see.",
    )
    p.add_argument(
        "--start-stall-secs", type=float, default=600.0,
        help="kill if process is alive but no log file exists OR log is "
             "still empty after this many seconds of monitor startup. "
             "Catches pre-step hangs (model init stuck, dataloader "
             "never producing first batch).",
    )
    p.add_argument("--alert-path", type=str, default="reports/alerts.jsonl")
    args = p.parse_args()

    log_path = Path(args.log)
    print(f"Safety watchdog: monitoring pid={args.pid}, log={log_path}, "
          f"check every {args.check_every}s")

    lm_window: deque[float] = deque(maxlen=50)
    last_pos = 0

    parse_errors = 0
    last_parse_error_log = 0.0
    monitor_start = time.time()
    last_log_progress = monitor_start                         # time we last saw a real progress record
    consecutive_explosions = 0                                # rolling count of in-a-row spikes
    while True:
        if not is_alive(args.pid):
            # The process is gone. Decide: clean completion or crash?
            completed, comp_rec = _has_completion_sentinel(log_path)
            if completed:
                print(f"Training process ended normally: {comp_rec}")
                Path(args.alert_path).parent.mkdir(parents=True, exist_ok=True)
                with open(args.alert_path, "a") as alert_f:
                    alert_f.write(json.dumps({
                        "time": time.time(),
                        "kind": "training_complete",
                        "pid": args.pid,
                        "completion_record": comp_rec,
                    }) + "\n")
                sys.exit(0)
            # No completion sentinel: training crashed or was killed
            # outside of this watchdog. Fire an alert so the operator sees
            # it instead of a silent success.
            Path(args.alert_path).parent.mkdir(parents=True, exist_ok=True)
            with open(args.alert_path, "a") as alert_f:
                alert_f.write(json.dumps({
                    "time": time.time(),
                    "kind": "training_crashed",
                    "pid": args.pid,
                    "log_path": str(log_path),
                    "hint": "process exited without writing a "
                            "training_complete sentinel. Check stderr "
                            "and the last log lines for the cause.",
                }) + "\n")
            print(f"!! Training pid={args.pid} ended WITHOUT completion sentinel "
                  f"-- treating as crash. See {args.alert_path}.")
            sys.exit(2)

        # Read new log lines
        if log_path.exists():
            try:
                size = log_path.stat().st_size
            except OSError:
                size = 0
            # Handle log rotation/truncation: if file shrunk below
            # last_pos, restart from start.
            if size < last_pos:
                print(f"  [safety] log shrunk ({size} < {last_pos}); restarting from 0")
                last_pos = 0
            with open(log_path, "r") as f:
                f.seek(last_pos)
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError as e:
                        parse_errors += 1
                        # Rate-limit parse-error chatter: log at most once every 60s.
                        if time.time() - last_parse_error_log > 60.0:
                            print(f"  [safety] parse error #{parse_errors}: "
                                  f"{e.msg} at col {e.colno} | line={line[:120]!r}")
                            last_parse_error_log = time.time()
                        continue
                    except Exception as e:                     # noqa: BLE001
                        parse_errors += 1
                        if time.time() - last_parse_error_log > 60.0:
                            print(f"  [safety] non-JSON error #{parse_errors}: "
                                  f"{type(e).__name__}: {e}")
                            last_parse_error_log = time.time()
                        continue
                    # Any record from train_loop counts as progress -- LM step
                    # records AND consolidation events both mean the trainer
                    # is alive. Records without lm_loss (consolidation, etc)
                    # bump the progress marker but skip the NaN/explosion
                    # checks below.
                    if rec.get("event") == "consolidation":
                        last_log_progress = time.time()
                        continue
                    if "lm_loss" not in rec:
                        continue
                    last_log_progress = time.time()
                    lm = rec["lm_loss"]
                    if math.isnan(lm) or math.isinf(lm):
                        kill_training(args.pid, f"NaN/Inf LM loss at step {rec.get('step')}",
                                      args.alert_path)
                        sys.exit(2)
                    if len(lm_window) >= 10:
                        mean = sum(lm_window) / len(lm_window)
                        if lm > mean * args.lm_explode_factor:
                            consecutive_explosions += 1
                            if consecutive_explosions >= args.lm_explode_consecutive:
                                kill_training(
                                    args.pid,
                                    f"LM loss exploded {consecutive_explosions} "
                                    f"chunks in a row (last: {lm:.2f} > "
                                    f"{args.lm_explode_factor}x trailing mean "
                                    f"{mean:.2f}) at step {rec.get('step')}",
                                    args.alert_path,
                                )
                                sys.exit(2)
                        else:
                            consecutive_explosions = 0        # reset the streak
                    lm_window.append(lm)
                last_pos = f.tell()

        # Disk space check
        free_gb = shutil.disk_usage(str(log_path.parent.parent)).free / 1e9
        if free_gb < args.min_free_gb:
            kill_training(args.pid, f"disk space critical: {free_gb:.1f} GB free",
                          args.alert_path)
            sys.exit(2)

        # Stall watchdog: process is alive but no new progress record.
        # Distinguish two states:
        #   (a) No log lines ever observed AND >start-stall-secs since
        #       monitor start -> training never produced its first record
        #       (model init stuck, dataloader deadlock at startup).
        #   (b) Log has progressed before but no new line in stall-secs
        #       -> training got stuck mid-run (collective hang, infinite
        #       inner loop, swap thrashing to a halt).
        # NaN/explosion checks cannot catch either because they need a
        # new log line to inspect.
        now = time.time()
        secs_since_progress = now - last_log_progress
        log_ever_has_lines = log_path.exists() and log_path.stat().st_size > 0
        if not log_ever_has_lines and (now - monitor_start) > args.start_stall_secs:
            kill_training(
                args.pid,
                f"training produced no log records in "
                f"{now - monitor_start:.0f}s since monitor start "
                f"(threshold {args.start_stall_secs:.0f}s). Likely model "
                f"init stuck or dataloader deadlocked before step 0.",
                args.alert_path,
            )
            sys.exit(2)
        elif log_ever_has_lines and secs_since_progress > args.stall_secs:
            kill_training(
                args.pid,
                f"training stalled: no new progress record for "
                f"{secs_since_progress:.0f}s (threshold "
                f"{args.stall_secs:.0f}s). Process still alive but log "
                f"is not advancing. Likely a collective hang, "
                f"dataloader deadlock, or swap thrashing.",
                args.alert_path,
            )
            sys.exit(2)

        time.sleep(args.check_every)


if __name__ == "__main__":
    main()
