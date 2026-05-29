from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_manifest(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": args.run_id,
        "run_root": args.run_root,
        "script": args.script,
        "status": args.status,
        "directories": {
            "data": f"{args.run_root}/data",
            "logs": f"{args.run_root}/logs",
            "checkpoints": f"{args.run_root}/checkpoints",
            "reports": f"{args.run_root}/reports",
        },
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-Model run context helpers.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    manifest = sub.add_parser("manifest")
    manifest.add_argument("--run-id", required=True)
    manifest.add_argument("--run-root", required=True)
    manifest.add_argument("--script", required=True)
    manifest.add_argument("--status", default="done")
    manifest.add_argument("--out", required=True)
    manifest.set_defaults(func=write_manifest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
