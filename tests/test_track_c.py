from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from collections import defaultdict
from pathlib import Path

from eval.track_c.baselines import rule_oracle_rows
from eval.track_c.generate import build_split_records, predicate_signature_for_action, render_model_prompt
from eval.track_c.metrics import state_accuracy
from eval.track_c.run_track_c import condition_inputs
from eval.track_c.training_data import build_records as build_training_records


ROOT = Path(__file__).resolve().parents[1]


def test_track_c_generator_is_deterministic_and_letter_balanced():
    first = [r.to_dict() for r in build_split_records(seed=0, split="train", n=200)]
    second = [r.to_dict() for r in build_split_records(seed=0, split="train", n=200)]
    assert first == second
    counts = Counter(r["gold_letter"] for r in first)
    assert set(counts) == {"A", "B", "C", "D"}
    assert max(counts.values()) / sum(counts.values()) <= 0.30


def test_track_c_groups_keep_visible_inputs_byte_identical():
    rows = [r.to_dict() for r in build_split_records(seed=0, split="standard_test", n=80)]
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    paired = [group for group in groups.values() if len(group) > 1]
    assert paired
    assert any(len({row["tau_seconds"] for row in group}) > 1 for group in paired)
    for group in paired[:10]:
        prompts = {render_model_prompt(row) for row in group}
        states = {json.dumps(row["hidden_state_json"], sort_keys=True) for row in group}
        choices = {json.dumps(row["choices"], sort_keys=True) for row in group}
        assert len(prompts) == 1
        assert len(states) == 1
        assert len(choices) == 1


def test_track_c_hidden_prompt_has_no_elapsed_timestamp_text():
    row = build_split_records(seed=1, split="standard_test", n=1)[0].to_dict()
    prompt = render_model_prompt(row)
    assert "Elapsed time since the session started" not in prompt
    assert str(row["tau_seconds"]) not in row["visible_prompt"]


def test_track_c_prompt_timestamp_adds_time_text():
    row = build_split_records(seed=1, split="standard_test", n=1)[0].to_dict()
    prompt = render_model_prompt(row, elapsed_seconds_text=row["tau_seconds"])
    assert "Elapsed time since the session started" in prompt


def test_track_c_condition_inputs_zero_and_shuffle_tau():
    rows = [r.to_dict() for r in build_split_records(seed=1, split="standard_test", n=20)]
    prompt, tau_forward, tau_prompt, shuffled_tau, alternate_gold = condition_inputs(rows[0], rows, 0, "prompt_timestamp")
    assert tau_forward == 0
    assert tau_prompt == rows[0]["tau_seconds"]
    assert shuffled_tau is None
    assert alternate_gold is None
    assert "Elapsed time since the session started" in prompt

    _, tau_forward, tau_prompt, shuffled_tau, alternate_gold = condition_inputs(rows[0], rows, 0, "shuffled_time_control")
    assert tau_forward == shuffled_tau
    assert tau_forward != rows[0]["tau_seconds"]
    assert tau_prompt is None
    assert alternate_gold is not None


def test_track_c_rule_oracle_is_perfect():
    rows = [r.to_dict() for r in build_split_records(seed=2, split="heldout_composition", n=200)]
    preds = rule_oracle_rows(rows)
    assert preds
    assert all(p["action"] == p["gold_action"] for p in preds)


def test_track_c_split_constraints_and_family_mix():
    train = [r.to_dict() for r in build_split_records(seed=3, split="train", n=1000)]
    fam = Counter(r["family"] for r in train)
    assert fam["cache"] + fam["job"] <= 0.20 * len(train)
    assert fam["multi"] >= 0.35 * len(train)
    assert "quota" not in fam
    val = [r.to_dict() for r in build_split_records(seed=3, split="val", n=500)]
    assert "quota" not in {r["family"] for r in val}
    standard = [r.to_dict() for r in build_split_records(seed=3, split="standard_test", n=500)]
    assert "quota" not in {r["family"] for r in standard}
    comp = [r.to_dict() for r in build_split_records(seed=3, split="heldout_composition", n=200)]
    assert {r["num_constraints"] for r in comp} <= {3, 4}
    heldout_family = [r.to_dict() for r in build_split_records(seed=3, split="heldout_family", n=200)]
    assert {r["family"] for r in heldout_family} == {"quota"}


def test_track_c_training_records_use_hidden_tau_or_prompt_tau():
    row = build_split_records(seed=4, split="train", n=1)[0].to_dict()
    hidden = build_training_records([row], condition="hidden_only")[0]
    prompt = build_training_records([row], condition="prompt_timestamp")[0]
    assert hidden["track"] == "track_c"
    assert hidden["tau_t"] == float(row["tau_seconds"])
    assert prompt["tau_t"] == 0.0
    assert "Elapsed time since the session started" in prompt["prefix_text"]
    assert hidden["mode"] == "track_c_hidden_only"
    assert prompt["mode"] == "track_c_prompt_timestamp"


def test_track_c_state_accuracy_uses_predicates_not_action_alias():
    row = build_split_records(seed=5, split="standard_test", n=20)[0].to_dict()
    wrong_action = next(action for action in row["choice_actions"].values() if action != row["gold_action"])
    pred = {
        "action": wrong_action,
        "gold_action": row["gold_action"],
        "gold_predicates": row["gold_predicates"],
        "predicted_predicates": predicate_signature_for_action(
            row["family"],
            row["hidden_state_json"],
            int(row["tau_seconds"]),
            row.get("active_constraints") or [],
            wrong_action,
        ),
    }
    acc, n = state_accuracy([pred])
    assert n == 1
    assert acc == 0.0


def test_track_c_generator_cli_requires_output_scope():
    proc = subprocess.run(
        [sys.executable, "-m", "eval.track_c.generate", "--seed", "0", "--profile", "safe"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert "output scope required" in (proc.stderr + proc.stdout)


def test_track_c_generator_cli_writes_split(tmp_path: Path):
    out_dir = tmp_path / "track_c"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.track_c.generate",
            "--out-dir",
            str(out_dir),
            "--seed",
            "0",
            "--profile",
            "safe",
            "--split",
            "standard_test",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(line) for line in (out_dir / "track_c_seed0_standard_test.jsonl").read_text().splitlines()]
    assert len(rows) == 1000
    assert {r["track"] for r in rows} == {"C"}
