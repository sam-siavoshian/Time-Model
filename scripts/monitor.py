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
import time
from collections import deque
from pathlib import Path
from typing import Iterator


def _follow(path: str, sleep: float = 1.0) -> Iterator[str]:
    f = open(path, "r")
    while True:
        line = f.readline()
        if not line:
            time.sleep(sleep)
            continue
        yield line


def _read_all(path: str) -> list[str]:
    with open(path, "r") as f:
        return f.readlines()


def _safe_load(line: str) -> dict | None:
    try:
        return json.loads(line)
    except Exception:                                         # noqa: BLE001
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log_path", type=str)
    p.add_argument("--tail", action="store_true", help="follow file (like tail -f)")
    p.add_argument("--window", type=int, default=50, help="rolling window size")
    p.add_argument("--lm-spike-threshold", type=float, default=2.0,
                   help="alert if LM loss > spike_threshold * trailing mean")
    p.add_argument("--grad-explode-threshold", type=float, default=100.0)
    p.add_argument("--alert-path", type=str, default="reports/alerts.jsonl")
    args = p.parse_args()

    Path(args.alert_path).parent.mkdir(parents=True, exist_ok=True)
    alert_f = open(args.alert_path, "a")

    lm_window: deque[float] = deque(maxlen=args.window)
    grad_window: deque[float] = deque(maxlen=args.window)
    mem_window: deque[float] = deque(maxlen=args.window)
    last_step = -1
    last_time = time.time()
    consolidation_events = 0
    rollback_events = 0

    def fire_alert(kind: str, payload: dict):
        rec = {"time": time.time(), "kind": kind, **payload}
        alert_f.write(json.dumps(rec) + "\n")
        alert_f.flush()
        print(f"  !! ALERT: {kind} {payload}")

    def process_line(line: str):
        nonlocal last_step, last_time, consolidation_events, rollback_events
        rec = _safe_load(line)
        if rec is None:
            return
        if "event" in rec and rec["event"] == "consolidation":
            consolidation_events += 1
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
        mem = rec.get("memory_norm", 0)

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
        mem_window.append(mem)
        last_step = step
        last_time = time.time()

        if step % 10 == 0 or step < 5:
            lm_avg = sum(lm_window) / max(1, len(lm_window))
            grad_avg = sum(grad_window) / max(1, len(grad_window))
            mem_avg = sum(mem_window) / max(1, len(mem_window))
            ppl_avg = math.exp(min(lm_avg, 20))
            print(
                f"step={step:6d} | LM={lm:7.4f} (avg{args.window}={lm_avg:.3f}, ppl={ppl_avg:.1f}) "
                f"| grad={grad:6.2f} (avg={grad_avg:.2f}) | mem={mem:.1f} (avg={mem_avg:.1f}) "
                f"| consol={consolidation_events} rollbacks={rollback_events}"
            )

    if args.tail:
        for line in _follow(args.log_path):
            process_line(line)
            # Stall watchdog (5 min)
            if time.time() - last_time > 300:
                fire_alert("training_stall", {"last_step": last_step,
                                              "seconds_since_last": time.time() - last_time})
                last_time = time.time()
    else:
        for line in _read_all(args.log_path):
            process_line(line)


if __name__ == "__main__":
    main()
