#!/usr/bin/env python3
"""Run one approved light command through the project compute guard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lean_reward_hacking.safety import run_guarded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("test", "smoke", "analysis", "plot", "verify"), default="verify")
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--memory-gb", type=float, default=4.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error("a command is required")
    try:
        result = run_guarded(
            args.command,
            operation_kind=args.kind,
            requested_cores=2,
            requested_memory_gb=args.memory_gb,
            requested_seconds=args.seconds,
            cwd=ROOT,
        )
    except subprocess.CalledProcessError as exc:
        if exc.output:
            sys.stdout.write(str(exc.output))
        if exc.stderr:
            sys.stderr.write(str(exc.stderr))
        return int(exc.returncode)
    print(json.dumps(result.__dict__, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
