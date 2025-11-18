"""Command-line interface for Thinker."""

from __future__ import annotations

import argparse
import json
import importlib.metadata as importlib_metadata
import platform
import sys
from pathlib import Path
from typing import Optional

from . import __version__ as THINKER_VERSION
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

    subparsers.add_parser("info", help="Show pipeline and environment details.")
    manifest_parser = subparsers.add_parser("manifest", help="Show latest Tinker adapter manifest.")
    manifest_parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Optional manifest path override. Defaults to evaluation.tinker_manifest_path or runs/latest_tinker_adapter.json.",
    )

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


def _package_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "not installed"


def _format_path(value: Path | None) -> str:
    if value is None:
        return "-"
    return str(value)


def _format_args(values: list[str] | None) -> str:
    if not values:
        return "-"
    return " ".join(values)


def _status_label(enabled: bool | None) -> str:
    return "enabled" if enabled else "disabled"


def _print_info(config_path: Path, config: PipelineConfig) -> None:
    """Emit a structured snapshot of the pipeline + runtime environment."""
    python_version = sys.version.split()[0]
    print("[thinker] environment")
    print(f"  Thinker version : {THINKER_VERSION}")
    print(f"  Python version  : {python_version}")
    print(f"  Platform        : {platform.platform()}")
    print(f"  Tinker SDK      : {_package_version('tinker')}")
    print(f"  Config file     : {config_path}")
    print(f"  Project root    : {config.project_root}")

    print("\n[tests]")
    tests_cfg = config.tests
    if tests_cfg:
        print(f"  status          : {_status_label(tests_cfg.enabled)}")
        print(f"  path            : {tests_cfg.path}")
        print(f"  markers         : {tests_cfg.markers or '-'}")
        print(f"  extra args      : {_format_args(tests_cfg.args)}")
    else:
        print("  status          : not configured")

    print("\n[data_validation]")
    val_cfg = config.data_validation
    if val_cfg:
        print(f"  status          : {_status_label(val_cfg.enabled)}")
        print(f"  dataset path    : {val_cfg.path}")
        print(f"  evidence mode   : {val_cfg.evidence_mode}")
        print(f"  embedding model : {val_cfg.embedding_model}")
        print(f"  similarity thr. : {val_cfg.similarity_threshold}")
        print(f"  claims path     : {_format_path(val_cfg.claims_path)}")
        print(f"  corpus path     : {_format_path(val_cfg.corpus_path)}")
    else:
        print("  status          : not configured")

    print("\n[training]")
    train_cfg = config.training
    if train_cfg:
        print(f"  status          : {_status_label(train_cfg.enabled)}")
        print(f"  backend         : {train_cfg.backend}")
        print(f"  config path     : {train_cfg.config_path}")
        print(f"  tinker config   : {_format_path(train_cfg.tinker_config_path)}")
        print(f"  tinker script   : {_format_path(train_cfg.tinker_script)}")
        print(f"  log dir         : {_format_path(train_cfg.log_dir)}")
    else:
        print("  status          : not configured")

    print("\n[evaluation]")
    eval_cfg = config.evaluation
    if eval_cfg:
        print(f"  status          : {_status_label(eval_cfg.enabled)}")
        print(f"  backend         : {eval_cfg.backend}")
        print(f"  base model      : {eval_cfg.base_model or '-'}")
        print(f"  claims file     : {eval_cfg.claims_file}")
        print(f"  corpus file     : {eval_cfg.corpus_file}")
        print(f"  checkpoint dir  : {_format_path(eval_cfg.checkpoint_dir)}")
        print(f"  output path     : {eval_cfg.output_path}")
        print(f"  max samples     : {eval_cfg.max_samples}")
        print(f"  tinker manifest : {_format_path(eval_cfg.tinker_manifest_path)}")
        print(f"  adapter name    : {eval_cfg.tinker_adapter_name or '-'}")
        print(f"  adapter path    : {eval_cfg.tinker_adapter_path or '-'}")
        print(f"  tinker tokens   : {eval_cfg.tinker_max_tokens}")
        print(f"  temperature     : {eval_cfg.tinker_temperature}")
    else:
        print("  status          : not configured")


def _resolve_manifest_path(config_path: Path, config: PipelineConfig, override: Path | None) -> Path:
    if override is not None:
        return override.resolve()
    eval_cfg = config.evaluation
    if eval_cfg and eval_cfg.tinker_manifest_path:
        return Path(eval_cfg.tinker_manifest_path)
    resolved = config_path.resolve()
    parents = resolved.parents
    try:
        repo_root = parents[2]
    except IndexError:
        repo_root = resolved.parent
    return (repo_root / "runs" / "latest_tinker_adapter.json").resolve()


def _print_manifest(manifest_path: Path) -> None:
    print("[thinker] adapter manifest")
    print(f"  path            : {manifest_path}")
    if not manifest_path.exists():
        print("  status          : not found")
        return
    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"  status          : invalid JSON ({exc})")
        return

    adapter_name = manifest.get("adapter_name") or "(unnamed)"
    adapter_path = manifest.get("adapter_path") or "-"
    base_model = manifest.get("base_model") or "-"
    config_path = manifest.get("config") or "-"
    timestamp = manifest.get("timestamp") or "-"

    print(f"  adapter name    : {adapter_name}")
    print(f"  adapter path    : {adapter_path}")
    print(f"  base model      : {base_model}")
    print(f"  config          : {config_path}")
    print(f"  timestamp       : {timestamp}")

    known_keys = {"adapter_name", "adapter_path", "base_model", "config", "timestamp"}
    extra = {k: v for k, v in manifest.items() if k not in known_keys}
    if extra:
        print("  extra fields    :")
        for key in sorted(extra):
            print(f"    {key}: {extra[key]}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    pipeline = _load_pipeline(args.config)

    try:
        if args.command == "info":
            _print_info(args.config, pipeline.config)
        elif args.command == "manifest":
            manifest_path = _resolve_manifest_path(args.config, pipeline.config, args.path)
            _print_manifest(manifest_path)
        elif args.command == "validate":
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
