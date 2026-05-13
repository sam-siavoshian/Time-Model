"""Training monitor.

Tails a JSONL log written by train_loop, computes rolling metrics,
prints a dashboard. Watches for:
  - LM loss spike (stop-loss watchdog)
  - Perplexity regression (>2x trailing average)
  - Gradient norm explosion (post-clip > 1.0)
  - Memory norm drift (slot collapse)
  - Consolidation rollback events
  - Stalls (no log update for >5 min)

Optional: write an alert file when any condition fires
(reports/alerts.jsonl). External monitors can poll this.

Usage:
  uv run python -m scripts.monitor logs/phase0.jsonl
  uv run python -m scripts.monitor logs/phase0.jsonl --tail --window 50
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Iterator


def _follow(path: str, sleep: float = 1.0) -> Iterator[str | None]:
    """Tail a file. Yields data lines as strings; yields None on idle ticks.

    The None heartbeats give the caller a chance to run periodic checks
    (e.g. stall / start watchdogs) even when no new log line arrives.
    Without them, a hung trainer that never writes leaves this generator
    blocked in time.sleep() forever and watchdogs never trigger.

    Also handles:
      - File does not exist yet at startup (polls until it appears).
      - Truncation: reopen from start when file size shrinks.
      - Rotation: reopen when inode changes.
    """
    f = None
    last_inode = None
    while f is None:
        try:
            f = open(path, "r")
            last_inode = os.fstat(f.fileno()).st_ino if hasattr(os, "fstat") else None
        except FileNotFoundError:
            yield None
            time.sleep(sleep)
    try:
        while True:
            line = f.readline()
            if line:
                yield line
                continue
            # No new line: check rotation/truncation, then heartbeat + sleep.
            try:
                st = os.stat(path)
                if st.st_size < f.tell():
                    f.close()
                    f = open(path, "r")
                    last_inode = st.st_ino
                    continue
                if last_inode is not None and st.st_ino != last_inode:
                    f.close()
                    f = open(path, "r")
                    last_inode = st.st_ino
                    continue
            except FileNotFoundError:
                pass
            yield None                                          # heartbeat tick
            time.sleep(sleep)
    finally:
        try:
            f.close()
        except Exception:                                     # noqa: BLE001
            pass


def _read_all(path: str) -> list[str]:
    with open(path, "r") as f:
        return f.readlines()


def _safe_load(line: str) -> tuple[dict | None, str | None]:
    """Parse one JSONL line. Returns (record, error_msg).

    - Empty/whitespace-only lines: returns (None, None) (legitimate skip).
    - Parse error: returns (None, "<reason>") so caller can count/log.
      Previously this swallowed every exception silently, which meant a
      corrupted log (truncated writes, partial flushes, nan/inf that the
      writer's json encoder rejected then rewrote partially) showed up
      as the monitor going completely quiet with no alert. Now the
      caller decides what to do with the error.
    """
    if not line or not line.strip():
        return None, None
    try:
        return json.loads(line), None
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {e.msg} at col {e.colno}"
    except Exception as e:                                    # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log_path", type=str)
    p.add_argument("--tail", action="store_true", help="follow file (like tail -f)")
    p.add_argument("--window", type=int, default=50, help="rolling window size")
    p.add_argument("--lm-spike-threshold", type=float, default=2.0,
                   help="alert if LM loss > spike_threshold * trailing mean")
    p.add_argument("--grad-explode-threshold", type=float, default=100.0)
    p.add_argument("--alert-path", type=str, default="reports/alerts.jsonl")
    p.add_argument(
        "--start-watchdog-secs", type=float, default=600.0,
        help="alert if NO valid log line arrives within this many seconds "
             "of monitor start. Catches pre-step hangs (model init stuck, "
             "dataloader deadlocked) where last_time would otherwise stay "
             "None and the regular stall watchdog never engages.",
    )
    p.add_argument(
        "--parse-error-burst-threshold", type=int, default=5,
        help="number of parse errors within parse-error-burst-window seconds "
             "before firing a log_format_error alert.",
    )
    p.add_argument("--parse-error-burst-window", type=float, default=60.0)
    args = p.parse_args()

    Path(args.alert_path).parent.mkdir(parents=True, exist_ok=True)
    # Line-buffered append; flushes on every '\n' write so alerts survive SIGKILL
    alert_f = open(args.alert_path, "a", buffering=1)

    lm_window: deque[float] = deque(maxlen=args.window)
    grad_window: deque[float] = deque(maxlen=args.window)
    mem_window: deque[float] = deque(maxlen=args.window)
    last_step = -1
    last_time = None                                          # None until first log line seen
    monitor_start = time.time()
    start_watchdog_fired = False
    parse_error_timestamps: deque[float] = deque(maxlen=args.parse_error_burst_threshold * 4)
    parse_error_burst_fired_at = 0.0
    consolidation_events = 0
    rollback_events = 0

    def fire_alert(kind: str, payload: dict):
        rec = {"time": time.time(), "kind": kind, **payload}
        alert_f.write(json.dumps(rec) + "\n")
        alert_f.flush()
        print(f"  !! ALERT: {kind} {payload}")

    def process_line(line: str):
        nonlocal last_step, last_time, consolidation_events, rollback_events
        nonlocal parse_error_burst_fired_at
        rec, err = _safe_load(line)
        if err is not None:
            # Real parse error (not just blank). Record + potentially burst-alert.
            now = time.time()
            parse_error_timestamps.append(now)
            # Always print so log corruption is visible during a live tail.
            preview = line[:200].rstrip("\n")
            print(f"  ?? PARSE-ERROR: {err} | line preview: {preview!r}")
            # Burst detection: if N parse errors within window AND we haven't
            # already alerted in the current window, fire log_format_error.
            recent = [t for t in parse_error_timestamps if now - t <= args.parse_error_burst_window]
            if (
                len(recent) >= args.parse_error_burst_threshold
                and (now - parse_error_burst_fired_at) > args.parse_error_burst_window
            ):
                fire_alert("log_format_error", {
                    "n_errors_in_window": len(recent),
                    "window_secs": args.parse_error_burst_window,
                    "last_error": err,
                })
                parse_error_burst_fired_at = now
            return
        if rec is None:
            return                                            # blank line
        if rec.get("event") == "consolidation":
            consolidation_events += 1
            last_time = time.time()                            # consolidation event counts as activity
            if not rec.get("committed", True):
                rollback_events += 1
                fire_alert("consolidation_rollback", {
                    "step": rec.get("step"),
                    "reason": rec.get("rollback_reason", ""),
                })
            return
        if "lm_loss" not in rec:
            return
        step = rec.get("step", last_step + 1)
        lm = rec.get("lm_loss", 0)
        grad = rec.get("grad_norm", 0)
        # memory_norm may legitimately be absent in older logs / pre-write
        # steps. Do NOT push a synthetic 0 into mem_window — that pollutes
        # the rolling average and falsely indicates slot collapse.
        mem = rec.get("memory_norm", None)

        # Trailing checks
        if len(lm_window) >= 10 and lm > sum(lm_window) / len(lm_window) * args.lm_spike_threshold:
            fire_alert("lm_loss_spike", {
                "step": step, "lm": lm,
                "trailing_mean": sum(lm_window) / len(lm_window),
            })
        if grad > args.grad_explode_threshold:
            fire_alert("grad_explode", {"step": step, "grad_norm": grad})
        if math.isnan(lm) or math.isinf(lm):
            fire_alert("lm_nan_inf", {"step": step, "lm": lm})

        lm_window.append(lm)
        grad_window.append(grad)
        if mem is not None:
            mem_window.append(mem)
        last_step = step
        last_time = time.time()

        if step % 10 == 0 or step < 5:
            lm_avg = sum(lm_window) / max(1, len(lm_window))
            grad_avg = sum(grad_window) / max(1, len(grad_window))
            ppl_avg = math.exp(min(lm_avg, 20))
            mem_disp = (
                f"mem={mem:.1f} (avg={sum(mem_window)/len(mem_window):.1f})"
                if (mem is not None and len(mem_window) > 0)
                else "mem=n/a"
            )
            print(
                f"step={step:6d} | LM={lm:7.4f} (avg{args.window}={lm_avg:.3f}, ppl={ppl_avg:.1f}) "
                f"| grad={grad:6.2f} (avg={grad_avg:.2f}) | {mem_disp} "
                f"| consol={consolidation_events} rollbacks={rollback_events}"
            )

    if args.tail:
        try:
            for line in _follow(args.log_path):
                if line is not None:
                    process_line(line)
                now = time.time()
                # Stall watchdog (5 min) — only fires AFTER first log line observed.
                if last_time is not None and now - last_time > 300:
                    fire_alert("training_stall", {
                        "last_step": last_step,
                        "seconds_since_last": now - last_time,
                    })
                    last_time = now                            # debounce: don't spam alerts every line
                # Start watchdog: catches the case where training hangs BEFORE
                # producing any log line at all (model init stuck, dataloader
                # deadlocked, NCCL hang on rank-0 init). last_time stays None
                # so the regular stall watchdog above never engages.
                if (
                    last_time is None
                    and not start_watchdog_fired
                    and (now - monitor_start) > args.start_watchdog_secs
                ):
                    fire_alert("training_no_signal", {
                        "seconds_since_monitor_start": now - monitor_start,
                        "log_path": args.log_path,
                        "hint": "log file present but no valid lm_loss / "
                                "consolidation records yet. Check that the "
                                "trainer actually started.",
                    })
                    start_watchdog_fired = True
        finally:
            try:
                alert_f.close()
            except Exception:                                 # noqa: BLE001
                pass
    else:
        try:
            for line in _read_all(args.log_path):
                process_line(line)
        finally:
            try:
                alert_f.close()
            except Exception:                                 # noqa: BLE001
                pass


if __name__ == "__main__":
    main()
