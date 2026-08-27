"""Command-line boundary shared by local verification and Colab notebooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .bank_parity import run_bank_parity
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

    parity = commands.add_parser(
        "bank-parity",
        aliases=("parity",),
        help="run the fail-closed Colab scalar-versus-replica-bank parity gate",
    )
    parity.add_argument("--config", required=False, type=Path)
    parity.add_argument("--remote-root", required=True, type=Path)
    parity.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parity.add_argument("--architecture", choices=("toy", "generic"), default=None)
    parity.add_argument("--steps", type=int, default=5)
    parity.add_argument("--samples", type=int, default=7)
    parity.add_argument("--batch-size", type=int, default=3)
    parity.add_argument("--eval-pairs", type=int, default=8)
    parity.add_argument("--seed", type=int, default=20_260_826)

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
    if args.command in {"bank-parity", "parity"}:
        result = run_bank_parity(
            args.config,
            args.remote_root,
            device=args.device,
            architecture=args.architecture,
            steps=args.steps,
            samples=args.samples,
            batch_size=args.batch_size,
            eval_pairs=args.eval_pairs,
            seed=args.seed,
        )
        print(json.dumps(result, sort_keys=True, default=str))
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
