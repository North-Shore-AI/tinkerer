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
    evidence_mode: str = "schema"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.75
    claims_path: Optional[Path] = None
    corpus_path: Optional[Path] = None
    relation_field: Optional[str] = None
    require_relations: bool = False


@dataclass(frozen=True)
class LocalTrainingConfig:
    config_path: Path
    backend: str = "hf_peft"
    enabled: bool = True
    tinker_config_path: Optional[Path] = None
    tinker_script: Optional[Path] = None
    log_dir: Optional[Path] = None


@dataclass(frozen=True)
class EvaluationConfig:
    claims_file: Path
    corpus_file: Path
    backend: str = "hf_peft"
    base_model: Optional[str] = None
    checkpoint_dir: Optional[Path] = None
    max_samples: int = 50
    output_path: Path = Path("eval_results.jsonl")
    enabled: bool = True
    tinker_manifest_path: Optional[Path] = None
    tinker_adapter_name: Optional[str] = None
    tinker_adapter_path: Optional[str] = None
    tinker_max_tokens: int = 256
    tinker_temperature: float = 0.0


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
            evidence_mode=val_raw.get("evidence_mode", "schema"),
            embedding_model=val_raw.get(
                "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            similarity_threshold=val_raw.get("similarity_threshold", 0.75),
            claims_path=_resolve_optional_path(base_dir, val_raw.get("claims_json")),
            corpus_path=_resolve_optional_path(base_dir, val_raw.get("corpus_json")),
            relation_field=val_raw.get("relation_field"),
            require_relations=val_raw.get("require_relations", False),
        )

    training_cfg = None
    if "training" in raw and raw["training"]:
        train_raw = raw["training"]
        training_cfg = LocalTrainingConfig(
            config_path=_resolve_path(base_dir, train_raw["config_path"]),
            backend=train_raw.get("backend", "hf_peft"),
            enabled=train_raw.get("enabled", True),
            tinker_config_path=_resolve_optional_path(base_dir, train_raw.get("tinker_config_path")),
            tinker_script=_resolve_optional_path(base_dir, train_raw.get("tinker_script")),
            log_dir=_resolve_optional_path(base_dir, train_raw.get("log_dir")),
        )

    eval_cfg = None
    if "evaluation" in raw and raw["evaluation"]:
        eval_raw = raw["evaluation"]
        eval_cfg = EvaluationConfig(
            backend=eval_raw.get("backend", "hf_peft"),
            base_model=eval_raw.get("base_model"),
            checkpoint_dir=_resolve_optional_path(base_dir, eval_raw.get("checkpoint_dir")),
            claims_file=_resolve_path(base_dir, eval_raw["claims_file"]),
            corpus_file=_resolve_path(base_dir, eval_raw["corpus_file"]),
            max_samples=eval_raw.get("max_samples", 50),
            output_path=_resolve_path(base_dir, eval_raw.get("output_path", "eval_results.jsonl")),
            enabled=eval_raw.get("enabled", True),
            tinker_manifest_path=_resolve_optional_path(base_dir, eval_raw.get("tinker_manifest_path")),
            tinker_adapter_name=eval_raw.get("tinker_adapter_name"),
            tinker_adapter_path=eval_raw.get("tinker_adapter_path"),
            tinker_max_tokens=eval_raw.get("tinker_max_tokens", 256),
            tinker_temperature=eval_raw.get("tinker_temperature", 0.0),
        )

    return PipelineConfig(
        project_root=base_dir,
        tests=tests_cfg,
        data_validation=validation_cfg,
        training=training_cfg,
        evaluation=eval_cfg,
    )
def _resolve_optional_path(base_dir: Path, raw: str | None) -> Path | None:
    if raw is None:
        return None
    return _resolve_path(base_dir, raw)
