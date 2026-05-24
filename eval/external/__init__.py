"""tau_sessions: an external benchmark for real-elapsed-time injection
into a frozen LLM.

  generate_tau_sessions.py   deterministic 300-session dataset builder
  eval_tau_bench.py          adapter-agnostic harness
  adapters/                  vanilla, prompt, CI implementations
  datasets/                  generated JSONL lives here

See README.md in this directory for usage.
License: MIT.
"""
