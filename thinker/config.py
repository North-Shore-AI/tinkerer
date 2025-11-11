"""Configuration models for the Thinker pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class SchemaField:
    """JSONL field requirement used by the dataset validator."""

    name: str
    type: str = "string"
    required: bool = True
    allow_empty: bool = False

    def validate(self, payload: Dict[str, Any]) -> tuple[bool, str | None]:
        if self.name not in payload:
            if self.required:
                return False, f"missing required field '{self.name}'"
            return True, None

        value = payload[self.name]
        if value is None:
            return (not self.required), (None if not self.required else f"field '{self.name}' is None")

        if self.type == "string":
            if not isinstance(value, str):
                return False, f"field '{self.name}' must be str, got {type(value).__name__}"
            if not value and not self.allow_empty:
                return False, f"field '{self.name}' cannot be empty"
        elif self.type == "array":
            if not isinstance(value, list):
                return False, f"field '{self.name}' must be list, got {type(value).__name__}"
            if not value and not self.allow_empty:
                return False, f"field '{self.name}' cannot be empty"
        elif self.type == "object":
            if not isinstance(value, dict):
                return False, f"field '{self.name}' must be dict, got {type(value).__name__}"
        else:
            return False, f"unsupported schema type '{self.type}'"

        return True, None


@dataclass(frozen=True)
class TestSuiteConfig:
    path: Path = Path("tests")
    markers: Optional[str] = None
    args: List[str] = field(default_factory=list)
    enabled: bool = True


@dataclass(frozen=True)
class DatasetValidationConfig:
    path: Path
    schema: List[SchemaField]
    max_examples: Optional[int] = None
    enabled: bool = True


@dataclass(frozen=True)
class LocalTrainingConfig:
    config_path: Path
    backend: str = "hf_peft"
    enabled: bool = True


@dataclass(frozen=True)
class EvaluationConfig:
    base_model: str
    checkpoint_dir: Path
    claims_file: Path
    corpus_file: Path
    max_samples: int = 50
    output_path: Path = Path("eval_results.jsonl")
    enabled: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    tests: Optional[TestSuiteConfig] = None
    data_validation: Optional[DatasetValidationConfig] = None
    training: Optional[LocalTrainingConfig] = None
    evaluation: Optional[EvaluationConfig] = None


def _resolve_path(base_dir: Path, raw: str | None) -> Path | None:
    if raw is None:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (base_dir / raw).resolve()
    return candidate


def _load_schema_fields(raw_fields: List[Dict[str, Any]]) -> List[SchemaField]:
    return [
        SchemaField(
            name=entry["name"],
            type=entry.get("type", "string"),
            required=entry.get("required", True),
            allow_empty=entry.get("allow_empty", False),
        )
        for entry in raw_fields
    ]


def load_pipeline_config(path: Path) -> PipelineConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    base_dir = config_path.parent

    tests_cfg = None
    if "tests" in raw and raw["tests"]:
        tests_raw = raw["tests"]
        tests_cfg = TestSuiteConfig(
            path=_resolve_path(base_dir, tests_raw.get("path", "tests")),
            markers=tests_raw.get("markers"),
            args=tests_raw.get("args", []),
            enabled=tests_raw.get("enabled", True),
        )

    validation_cfg = None
    if "data_validation" in raw and raw["data_validation"]:
        val_raw = raw["data_validation"]
        schema_fields = _load_schema_fields(val_raw.get("schema", []))
        validation_cfg = DatasetValidationConfig(
            path=_resolve_path(base_dir, val_raw["path"]),
            schema=schema_fields,
            max_examples=val_raw.get("max_examples"),
            enabled=val_raw.get("enabled", True),
        )

    training_cfg = None
    if "training" in raw and raw["training"]:
        train_raw = raw["training"]
        training_cfg = LocalTrainingConfig(
            config_path=_resolve_path(base_dir, train_raw["config_path"]),
            backend=train_raw.get("backend", "hf_peft"),
            enabled=train_raw.get("enabled", True),
        )

    eval_cfg = None
    if "evaluation" in raw and raw["evaluation"]:
        eval_raw = raw["evaluation"]
        eval_cfg = EvaluationConfig(
            base_model=eval_raw.get("base_model", "meta-llama/Llama-3.1-8B-Instruct"),
            checkpoint_dir=_resolve_path(base_dir, eval_raw["checkpoint_dir"]),
            claims_file=_resolve_path(base_dir, eval_raw["claims_file"]),
            corpus_file=_resolve_path(base_dir, eval_raw["corpus_file"]),
            max_samples=eval_raw.get("max_samples", 50),
            output_path=_resolve_path(base_dir, eval_raw.get("output_path", "eval_results.jsonl")),
            enabled=eval_raw.get("enabled", True),
        )

    return PipelineConfig(
        project_root=base_dir,
        tests=tests_cfg,
        data_validation=validation_cfg,
        training=training_cfg,
        evaluation=eval_cfg,
    )
