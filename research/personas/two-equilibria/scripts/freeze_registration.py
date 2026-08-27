#!/usr/bin/env python3
"""Freeze and verify the source-bound registration used by Colab notebooks."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import time

import build_notebooks


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "REGISTRATION_FREEZE.json"
GOAL_SHA256 = "e5ab6b298bcad129ba00787e8fd0c28bad1374f2fa6c9ce066aaad037f6ffc69"
NOTEBOOKS = (
    "01_toy_sweep_colab.ipynb",
    "02_mlp_control_colab.ipynb",
    "03_perturbation_colab.ipynb",
    "04_analysis_export_colab.ipynb",
    "05_lm_workflow_colab.ipynb",
)
REGISTRATION_FILES = (
    "LEAN_REWARD_HACKING_GOAL.md",
    "reports/architecture_registration.md",
    "reports/statistical_methods.md",
    "reports/compute_manifest.md",
    "reports/literature_gap_audit.md",
    "reports/source_ledger.csv",
    "reports/claim_matrix.csv",
    "reports/LM_RESOURCE_REQUIREMENTS.json",
    "configs/toy_colab.toml",
    "configs/basin_colab.toml",
    "configs/generic_colab.toml",
    "configs/perturbation_colab.toml",
    "configs/lm_colab.toml",
    "configs/toy_smoke.toml",
    "requirements-colab.txt",
    "requirements-lm-colab.txt",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def notebook_source_identity(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in value.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    match = re.search(r'SOURCE_ARCHIVE_SHA256\s*=\s*"([0-9a-f]{64})"', source)
    archive_match = re.search(r'SOURCE_ARCHIVE_B64\s*=\s*"([A-Za-z0-9+/=]+)"', source)
    if match is None or archive_match is None:
        raise RuntimeError(f"notebook has no embedded source identity: {path}")
    payload = base64.b64decode(archive_match.group(1).encode("ascii"))
    if hashlib.sha256(payload).hexdigest() != match.group(1):
        raise RuntimeError(f"notebook embedded source hash mismatch: {path}")
    return match.group(1)


def current_records() -> dict[str, object]:
    missing = [name for name in REGISTRATION_FILES if not (ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"missing registration files: {missing}")
    snapshot = build_notebooks.source_snapshot()
    notebook_hashes = {
        name: sha256_file(ROOT / "notebooks" / name)
        for name in NOTEBOOKS
    }
    identities = {
        notebook_source_identity(ROOT / "notebooks" / name)
        for name in NOTEBOOKS
    }
    if identities != {snapshot["archive_sha256"]}:
        raise RuntimeError(
            "checked-in notebooks do not share the current registered source archive"
        )
    files = {name: sha256_file(ROOT / name) for name in REGISTRATION_FILES}
    if files["LEAN_REWARD_HACKING_GOAL.md"] != GOAL_SHA256:
        raise RuntimeError("protected goal hash changed")
    return {
        "schema_version": 1,
        "protected_goal_sha256": GOAL_SHA256,
        "source_commit": snapshot["commit"],
        "source_dirty": snapshot["dirty"],
        "source_archive_sha256": snapshot["archive_sha256"],
        "source_file_count": len(snapshot["files"]),
        "registration_files": files,
        "notebooks": notebook_hashes,
    }


def atomic_write_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    records = current_records()
    if args.verify:
        if not OUTPUT.is_file():
            raise RuntimeError(f"missing freeze artifact: {OUTPUT}")
        frozen = json.loads(OUTPUT.read_text(encoding="utf-8"))
        observed = {key: frozen.get(key) for key in records}
        if observed != records:
            raise RuntimeError("registration freeze differs from current registered inputs")
        print(f"verified {OUTPUT}")
        return 0
    value = {
        **records,
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "live_outputs_seen_before_freeze": False,
    }
    atomic_write_json(OUTPUT, value)
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
