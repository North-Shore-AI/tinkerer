"""Command-line interface for Thinker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from .config import PipelineConfig, load_pipeline_config
from .pipeline import ThinkerPipeline


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "pipeline_scifact.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thinker", description="Run TDD-first CNS pipelines.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="Pipeline config path (defaults to thinker/configs/pipeline_scifact.yaml).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate", help="Run pytest + dataset validation.")

    train_parser = subparsers.add_parser("train", help="Run training after validation.")
    train_parser.add_argument("--backend", default=None, help="Training backend override.")
    train_parser.add_argument(
        "--skip-validation", action="store_true", help="Skip validation (not recommended)."
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate a checkpoint.")
    eval_parser.add_argument(
        "--skip-validation", action="store_true", help="Skip validation (not recommended)."
    )

    run_parser = subparsers.add_parser("run", help="Validate, train, and evaluate.")
    run_parser.add_argument("--backend", default=None, help="Training backend override.")

    return parser


def _load_pipeline(path: Path) -> ThinkerPipeline:
    config: PipelineConfig = load_pipeline_config(path)
    return ThinkerPipeline(config)


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    pipeline = _load_pipeline(args.config)

    try:
        if args.command == "validate":
            pipeline.validate()
        elif args.command == "train":
            pipeline.train(backend=args.backend, skip_validation=args.skip_validation)
        elif args.command == "eval":
            pipeline.evaluate(skip_validation=args.skip_validation)
        elif args.command == "run":
            pipeline.run(backend=args.backend)
        else:
            parser.error("Unknown command")
            return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[thinker] error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
