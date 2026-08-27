"""Command-line boundary shared by local verification and Colab notebooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .campaign import analyze_path, export_compact, run_colab, tiny_validate
from .config import config_hash, load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lean-reward-hacking")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-config")
    validate.add_argument("--config", required=True, type=Path)

    tiny = commands.add_parser("tiny-validate")
    tiny.add_argument("--config", required=True, type=Path)
    tiny.add_argument("--remote-root", required=True, type=Path)

    run = commands.add_parser("colab-run")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--remote-root", required=True, type=Path)

    analyze = commands.add_parser("analyze")
    analyze.add_argument("--bundle", required=True, type=Path)

    export = commands.add_parser("export")
    export.add_argument("--remote-root", required=True, type=Path)
    export.add_argument("--local-bundle", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-config":
        config = load_config(args.config)
        print(json.dumps({"experiment": config.experiment, "config_sha256": config_hash(config)}, sort_keys=True))
        return 0
    if args.command == "tiny-validate":
        print(json.dumps(tiny_validate(args.config, args.remote_root), sort_keys=True, default=str))
        return 0
    if args.command == "colab-run":
        print(run_colab(args.config, args.remote_root))
        return 0
    if args.command == "analyze":
        print(json.dumps(analyze_path(args.bundle), sort_keys=True, default=str))
        return 0
    if args.command == "export":
        print(export_compact(args.remote_root, args.local_bundle))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
