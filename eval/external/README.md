# tau_sessions: external benchmark for real-elapsed-time injection

`tau_sessions` is the first public benchmark that asks an LLM to behave
differently as a function of wall-clock elapsed time (`tau`), provided
as a real tensor channel rather than as text. Existing time benchmarks
(TimeBench, TempReason, TimeQA, MenatQA) all serialize time into the
prompt and measure what the model knows about clocks in the abstract.
`tau_sessions` measures whether the model can be conditioned on time
the way a thermostat is conditioned on temperature: a separate, always-
on, continuous input.

The reference adapter (`ci`) implements Chronometric Injection from
[Time-Model](https://github.com/sam-siavoshian/Time-Model) (v15.0
release). Two baselines ship for comparison: `prompt` (tau injected as
"[elapsed: 3h 42m]" text) and `vanilla` (no tau at all).

License: MIT. Contributions welcome — see "How to contribute new
adapters" below.

---

## What the benchmark tests

300 synthetic sessions across 6 tau buckets (50 sessions each):
`1s, 60s, 600s, 6h, 24h, 7d`. Each session is one of three task types
(100 sessions each):

| Task              | What it measures                                           | Scoring         |
|-------------------|------------------------------------------------------------|-----------------|
| `duration_recall` | "How long since we started talking?" -> the model must read tau | log10-MAE on parsed seconds |
| `staleness`       | "Event lasts X minutes, is it still active?" -> compare tau to a duration | yes/no exact match |
| `adaptive`        | "You have until tau to answer, be appropriately concise." -> measure response length vs tau | Pearson r between log(tau) and log(length) |

The dataset is fully synthetic and regenerable byte-for-byte from a
seed, so reproducing or extending the benchmark needs no external
download.

---

## Quick start

```bash
# 1. Regenerate the dataset (or use the one checked in)
uv run python -m eval.external.generate_tau_sessions \
  --seed 42 \
  --out eval/external/datasets/tau_sessions.jsonl

# 2. Vanilla baseline (no tau, base Qwen only)
uv run python -m eval.external.eval_tau_bench \
  --adapter vanilla \
  --base Qwen/Qwen2.5-3B-Instruct \
  --device auto

# 3. Prompt baseline (tau injected as text)
uv run python -m eval.external.eval_tau_bench \
  --adapter prompt \
  --base Qwen/Qwen2.5-3B-Instruct \
  --device auto

# 4. Chronometric Injection (v15 release checkpoint)
curl -L -o seed0.pt \
  https://github.com/sam-siavoshian/Time-Model/releases/download/v15.0/qwen_time_v15s_20260523_141410_seed0.pt
uv run python -m eval.external.eval_tau_bench \
  --adapter ci \
  --base Qwen/Qwen2.5-3B-Instruct \
  --checkpoint seed0.pt \
  --device auto
```

Reports land at `reports/external_tau_bench_<adapter>.json` and contain
per-bucket metrics, bootstrap 95% CIs (B=1000), a composite score, and
the full per-session prediction rows so anyone can recompute.

For a fast end-to-end smoke (20 sessions on CPU, ~5 minutes):

```bash
uv run python -m eval.external.eval_tau_bench \
  --adapter vanilla --base Qwen/Qwen2.5-3B-Instruct --n 20
```

---

## Expected results

The CI paper's hypothesis: a frozen base LLM has no clock, so it cannot
do `duration_recall` or `staleness` without an extra channel. Telling
it in text closes most of the gap. Telling it via a chrono tensor
closes the rest while keeping the prompt clean. On `adaptive`, CI and
prompt-injection should be roughly comparable because the model needs
to do active length control, not just read tau.

Rough ordering we predict (and observe on v15s seed 0 in our internal
runs, see `reports/external_tau_bench_*.json` once you run them):

| Task              | vanilla     | prompt     | ci         |
|-------------------|-------------|------------|------------|
| `duration_recall` (log-MAE, lower = better) | very high   | low        | low        |
| `staleness` (accuracy)         | near chance | high       | high       |
| `adaptive` (Pearson r)         | ~0          | moderate   | comparable |

If `ci` does NOT beat `vanilla` on `duration_recall` and `staleness`,
the CI claim is falsified for that checkpoint. We pre-register this as
the load-bearing comparison. The `prompt` adapter is the realistic
upper-bound baseline.

---

## Devices and resources

- CPU: works end-to-end. ~3 minutes per adapter on Apple-Silicon CPU
  with 20 sessions (`--n 20`); ~45 minutes for the full 300.
- MPS (Apple Silicon GPU): ~10x faster than CPU for the base model.
  Set `--device mps`.
- CUDA: fastest. Full 300-session run is ~5-8 minutes per adapter on
  a single A100/H100/GB10.

Memory: the CI adapter holds one frozen Qwen-3B in bf16 (~7 GB) plus
the trainable params (38 MB checkpoint). Vanilla and prompt only need
the base model.

---

## How to contribute new adapters

Subclass `TauAdapter` in a new file under
`eval/external/adapters/<your_name>_adapter.py`:

```python
from .base import TauAdapter, greedy_generate

class MyAdapter(TauAdapter):
    name = "mine"

    def load(self):
        # populate self.model and self.tokenizer
        ...

    def generate(self, prompt: str, tau_seconds: float) -> str:
        # return decoded text for (prompt, tau)
        ...
```

Register it in `eval/external/adapters/__init__.py` by adding an entry
to `_build_registry()`. The harness will pick it up automatically.

A good adapter PR includes:
- the adapter class with a docstring explaining tau injection strategy
- a 20-session smoke comparison vs `vanilla` and `prompt` in the PR
  description
- a unit test under `tests/test_tau_bench.py` confirming the adapter
  loads (gated by `TAU_BENCH_SMOKE_HF=1` if it downloads weights)

---

## File layout

```
eval/external/
  README.md                       <- you are here
  __init__.py
  generate_tau_sessions.py        deterministic 300-session JSONL builder
  eval_tau_bench.py               adapter-agnostic harness + scoring
  datasets/
    tau_sessions.jsonl            seed 42 dataset (regenerable)
  adapters/
    __init__.py                   registry: vanilla, prompt, ci
    base.py                       TauAdapter ABC + greedy decoder
    vanilla_adapter.py            no tau channel
    prompt_adapter.py             tau injected as text prefix
    ci_adapter.py                 v15 LoRA + per-layer FiLM chrono
```

Internal sanity tests (T1-T5 on the CI checkpoint) live under
`model/qwen_time_check*.py`. They are not part of this external suite
because they assume direct access to the chrono-injected model
internals.

---

## License

MIT. See [LICENSE](../../LICENSE) at the repo root.
