from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "reports" / "current" / "claims.json"
SCHEMA = ROOT / "reports" / "current" / "schema.json"
MANIFEST = ROOT / "reports" / "current" / "manifest.json"
REPORT_EXTS = {".json", ".jsonl", ".md", ".txt", ".log"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def dump_json(data: Any) -> str:
    return json.dumps(data, indent=2) + "\n"


def iter_path_values(obj: Any, schema: dict[str, Any]):
    path_keys = set(schema["path_keys"])
    list_keys = set(schema["path_list_keys"])
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "original_path":
                continue
            if key in path_keys and isinstance(value, str):
                yield value
            elif key in list_keys and isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        yield item
            else:
                yield from iter_path_values(value, schema)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_path_values(item, schema)
    elif isinstance(obj, str):
        yield obj


def validate_schema(manifest: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in schema["required_top_level"]:
        if key not in manifest:
            errors.append(f"missing top-level key: {key}")
    statuses = set(schema["status_values"])

    def walk_status(obj: Any, path: str = "$") -> None:
        if isinstance(obj, dict):
            status = obj.get("status")
            if isinstance(status, str) and status not in statuses:
                errors.append(f"{path}.status has unsupported value {status!r}")
            for key, value in obj.items():
                walk_status(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for idx, value in enumerate(obj):
                walk_status(value, f"{path}[{idx}]")

    walk_status(manifest)
    return errors


def validate_paths(manifest: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for value in iter_path_values(manifest, schema):
        if value.startswith("http") or value == "None":
            continue
        if "*" in value:
            if not glob.glob(str(ROOT / value)):
                missing.append(value)
        elif Path(value).suffix in {".py", ".sh", ".json", ".md", ".tex", ".jsonl", ".txt", ".log"}:
            if not (ROOT / value).exists():
                missing.append(value)
    return missing


def mentioned_report_patterns(manifest: dict[str, Any], schema: dict[str, Any]) -> tuple[set[str], set[str]]:
    values = set()
    for value in iter_path_values(manifest, schema):
        if value.startswith("reports/"):
            values.add(value)
    exact = {value for value in values if "*" not in value}
    patterns = {value for value in values if "*" in value}
    return exact, patterns


def unclassified_root_reports(manifest: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    exact, patterns = mentioned_report_patterns(manifest, schema)
    out: list[str] = []
    for path in sorted((ROOT / "reports").glob("**/*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith("reports/archive/") or rel.startswith("reports/current/"):
            continue
        if path.suffix.lower() not in REPORT_EXTS:
            continue
        if rel in exact or any(fnmatch.fnmatch(rel, pattern) for pattern in patterns):
            continue
        out.append(rel)
    return out


def generate() -> dict[str, Any]:
    manifest = load_json(CLAIMS)
    schema = load_json(SCHEMA)
    errors = validate_schema(manifest, schema)
    missing = validate_paths(manifest, schema)
    unclassified = unclassified_root_reports(manifest, schema)
    if errors or missing or unclassified:
        details = []
        if errors:
            details.append("schema errors:\n" + "\n".join(f"  {x}" for x in errors))
        if missing:
            details.append("missing paths:\n" + "\n".join(f"  {x}" for x in missing))
        if unclassified:
            details.append("unclassified root reports:\n" + "\n".join(f"  {x}" for x in unclassified))
        raise SystemExit("\n\n".join(details))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out", default=str(MANIFEST))
    args = parser.parse_args()

    manifest = generate()
    rendered = dump_json(manifest)
    out = Path(args.out)
    if args.check:
        existing = out.read_text() if out.exists() else ""
        if existing != rendered:
            print(f"{out} is not generated from {CLAIMS}", file=sys.stderr)
            raise SystemExit(1)
        return
    out.write_text(rendered)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
