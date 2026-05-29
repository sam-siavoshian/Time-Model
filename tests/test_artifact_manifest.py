from __future__ import annotations

import glob
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _iter_path_values(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {
                "canonical_report",
                "baseline_report",
                "dataset",
                "generator",
                "manifest",
                "path",
                "successor",
            } and isinstance(value, str):
                yield value
            elif key in {
                "canonical_reports",
                "raw_inputs",
                "supporting_reports",
                "behavior_reports",
                "figure_inputs",
                "generators",
                "paper_locations",
                "include",
                "evidence",
            } and isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        yield item
            else:
                yield from _iter_path_values(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_path_values(item)


def test_current_manifest_is_valid_json_and_paths_resolve():
    manifest_path = ROOT / "reports" / "current" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    assert manifest["policy"]["moves_performed"] is True
    assert manifest["current"]["tps"]["canonical_report"] == "reports/tps/headline.json"

    missing = []
    for value in _iter_path_values(manifest):
        if value.startswith("http") or value == "None":
            continue
        if "*" in value:
            if not glob.glob(str(ROOT / value)):
                missing.append(value)
        elif value.endswith(".py") or value.endswith(".sh") or value.endswith(".json") or value.endswith(".md") or value.endswith(".tex") or value.endswith(".jsonl"):
            if not (ROOT / value).exists():
                missing.append(value)
    assert missing == []


def test_archive_manifests_exist():
    for path in [
        ROOT / "docs" / "history" / "MANIFEST.md",
        ROOT / "reports" / "archive" / "ipcn_memory" / "MANIFEST.md",
        ROOT / "reports" / "archive" / "model_versions_v10_v14" / "MANIFEST.md",
        ROOT / "reports" / "archive" / "superseded_prompt_baseline" / "MANIFEST.md",
        ROOT / "reports" / "archive" / "tpdr_v1" / "MANIFEST.md",
    ]:
        assert path.exists()


def test_current_manifest_is_generated_and_complete():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_current_manifest.py"), "--check"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
