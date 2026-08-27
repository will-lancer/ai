#!/usr/bin/env python3
"""Verify the local compact-result allowlist, schema, and checksums."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="verify_bundle.py")
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--allow-missing-manifest", action="store_true")
    parser.add_argument("--allow-missing-checksums", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # Import after argument parsing so --help works from a source checkout.
    source_root = Path(__file__).resolve().parents[1] / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from lean_reward_hacking.schemas import validate_compact_bundle

    problems = validate_compact_bundle(
        args.bundle,
        require_manifest=not args.allow_missing_manifest,
        require_checksums=not args.allow_missing_checksums,
    )
    if args.as_json:
        print(json.dumps({"bundle": str(args.bundle), "valid": not problems, "problems": problems}, indent=2, sort_keys=True))
    else:
        if problems:
            print(f"invalid compact bundle: {args.bundle}", file=sys.stderr)
            for problem in problems:
                print(f"- {problem}", file=sys.stderr)
        else:
            print(f"valid compact bundle: {args.bundle}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
