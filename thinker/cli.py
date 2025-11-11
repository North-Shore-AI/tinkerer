"""Command-line interface for Thinker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from .config import PipelineConfig, load_pipeline_config
from .pipeline import ThinkerPipeline
from .data import run_data_setup


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

    data_parser = subparsers.add_parser(
        "data",
        help="Data utilities (download, convert, validate).",
    )
    data_sub = data_parser.add_subparsers(dest="data_command", required=True)

    data_setup = data_sub.add_parser("setup", help="Download/prepare SciFact data.")
    data_setup.add_argument(
        "--dataset",
        choices=("scifact", "fever"),
        default="scifact",
        help="Which dataset to prepare.",
    )
    data_setup.add_argument(
        "--claims-json",
        type=Path,
        default=Path("cns-support-models/data/raw/scifact/claims_train.jsonl"),
        help="SciFact claims JSONL output path.",
    )
    data_setup.add_argument(
        "--corpus-json",
        type=Path,
        default=Path("cns-support-models/data/raw/scifact/corpus.jsonl"),
        help="SciFact corpus JSONL output path.",
    )
    data_setup.add_argument(
        "--output",
        type=Path,
        default=Path("cns-support-models/data/processed/scifact_claim_extractor.jsonl"),
        help="Processed SciFact dataset path.",
    )
    data_setup.add_argument(
        "--fever-claims",
        type=Path,
        default=Path("cns-support-models/data/raw/fever/train.jsonl"),
        help="FEVER claims JSONL path.",
    )
    data_setup.add_argument(
        "--fever-wiki-dir",
        type=Path,
        default=Path("cns-support-models/data/raw/fever/wiki-pages/wiki-pages"),
        help="FEVER wiki shards directory.",
    )
    data_setup.add_argument(
        "--fever-output",
        type=Path,
        default=Path("cns-support-models/data/processed/fever_claim_extractor.jsonl"),
        help="FEVER processed dataset path.",
    )
    data_setup.add_argument(
        "--fever-include-nei",
        action="store_true",
        help="Include NOT ENOUGH INFO claims when converting FEVER.",
    )
    data_setup.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip running dataset validation after conversion.",
    )
    data_setup.add_argument(
        "--validation-mode",
        choices=("exact", "embedding"),
        default="exact",
        help="Evidence validation mode when running validator.",
    )
    data_setup.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.75,
        help="Cosine similarity threshold for embedding validation.",
    )

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
            metrics = pipeline.evaluate(skip_validation=args.skip_validation)
            print(f"[thinker] eval metrics: {metrics}")
        elif args.command == "run":
            metrics = pipeline.run(backend=args.backend)
            print(f"[thinker] run metrics: {metrics}")
        elif args.command == "data":
            if args.data_command == "setup":
                run_data_setup(
                    dataset=args.dataset,
                    claims_path=args.claims_json,
                    corpus_path=args.corpus_json,
                    output_path=args.output,
                    fever_claims=args.fever_claims,
                    fever_wiki_dir=args.fever_wiki_dir,
                    fever_output=args.fever_output,
                    fever_include_nei=args.fever_include_nei,
                    skip_validation=args.skip_validation,
                    validation_mode=args.validation_mode,
                    similarity_threshold=args.similarity_threshold,
                )
            else:
                parser.error("Unknown data subcommand")
        else:
            parser.error("Unknown command")
            return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[thinker] error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
