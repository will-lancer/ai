#!/usr/bin/env python3
"""Bind a completed rendered review to the four-pass report audit ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def finalize(audit_path: Path, rendered_artifact: Path, notes: str) -> dict[str, object]:
    if not audit_path.is_file() or not rendered_artifact.is_file():
        raise FileNotFoundError(audit_path if not audit_path.is_file() else rendered_artifact)
    value = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("passes"), list):
        raise ValueError("invalid report audit ledger")
    passes = value["passes"]
    if len(passes) != 4 or any(not isinstance(item, Mapping) for item in passes):
        raise ValueError("four report-audit passes are required")
    if any(item.get("status") != "passed" for item in passes[:3]):
        raise ValueError("passes 1-3 must pass before rendered review")
    artifact_pass = passes[3]
    evidence = artifact_pass.get("evidence")
    if not isinstance(evidence, dict) or evidence.get("machine_artifacts_passed") is not True:
        raise ValueError("machine artifact verification has not passed")
    evidence["rendered_review"] = {
        "artifact": str(rendered_artifact),
        "sha256": sha256_file(rendered_artifact),
        "notes": notes,
        "reviewed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    artifact_pass["status"] = "passed"
    artifact_pass["result"] = "PASS"
    artifact_pass["unresolved"] = []
    artifact_pass.setdefault("checks", []).append("rendered artifact visually reviewed")
    value["status"] = "passed"
    value["overall_passed"] = True
    atomic_json(audit_path, value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--rendered-artifact", type=Path, required=True)
    parser.add_argument("--notes", required=True)
    args = parser.parse_args()
    finalize(args.audit, args.rendered_artifact, args.notes)
    print(f"finalized {args.audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
