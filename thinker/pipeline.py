"""High-level orchestration for Thinker workflows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

from .config import PipelineConfig
from .evaluation import Evaluator
from .training import TrainingReport, create_training_backend
from .validation import DatasetValidator, TestSuiteRunner


@dataclass
class PipelineState:
    validation_ran: bool = False
    training_completed: bool = False
    tinker_adapter_name: Optional[str] = None
    tinker_adapter_path: Optional[str] = None
    tinker_adapter_manifest: Optional[Path] = None
    tinker_base_model: Optional[str] = None


class ThinkerPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.state = PipelineState()

    def validate(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {"tests": None, "data_validation": None}
        tests_cfg = self.config.tests
        if tests_cfg and tests_cfg.enabled:
            TestSuiteRunner(tests_cfg).run()
            summary["tests"] = {
                "status": "passed",
                "path": str(tests_cfg.path),
                "markers": tests_cfg.markers,
                "args": tests_cfg.args,
            }
        elif tests_cfg:
            summary["tests"] = {
                "status": "skipped",
                "enabled": False,
                "path": str(tests_cfg.path),
            }

        validation_cfg = self.config.data_validation
        if validation_cfg and validation_cfg.enabled:
            result = DatasetValidator(validation_cfg).validate()
            if not result.is_valid:
                errors = "\n".join(result.errors[:10])
                raise ValueError(f"Dataset validation failed for {result.path}:\n{errors}")
            summary["data_validation"] = {
                "status": "passed",
                "path": str(result.path),
                "total_examples": result.total_examples,
                "evidence_mode": validation_cfg.evidence_mode,
                "similarity_threshold": validation_cfg.similarity_threshold,
                "schema_fields": len(validation_cfg.schema),
                "require_relations": validation_cfg.require_relations,
            }
        elif validation_cfg:
            summary["data_validation"] = {
                "status": "skipped",
                "enabled": False,
                "path": str(validation_cfg.path),
            }

        self.state.validation_ran = True
        return summary

    def train(self, backend: Optional[str] = None, skip_validation: bool = False) -> TrainingReport | None:
        if not skip_validation:
            self._ensure_validation()
        train_cfg = self.config.training
        if not train_cfg or not train_cfg.enabled:
            return None

        backend_cfg = train_cfg
        if backend is not None and backend != train_cfg.backend:
            backend_cfg = replace(train_cfg, backend=backend)
        trainer = create_training_backend(backend_cfg)
        report = trainer.train()
        self.state.training_completed = True
        if isinstance(report, TrainingReport) and report.backend == "tinker":
            self.state.tinker_adapter_name = report.metrics.get("adapter_name")
            adapter_path = report.metrics.get("adapter_path")
            self.state.tinker_adapter_path = adapter_path
            manifest_path = report.metrics.get("manifest_path")
            if manifest_path:
                self.state.tinker_adapter_manifest = Path(manifest_path)
            self.state.tinker_base_model = report.metrics.get("base_model")
        return report

    def evaluate(self, skip_validation: bool = False) -> dict:
        if not skip_validation:
            self._ensure_validation()
        eval_cfg = self.config.evaluation
        if not eval_cfg or not eval_cfg.enabled:
            return {}
        evaluator = Evaluator(eval_cfg, state=self.state)
        return evaluator.run()

    def run(self, backend: Optional[str] = None) -> dict:
        self.validate()
        self.train(backend=backend, skip_validation=True)
        return self.evaluate(skip_validation=True)

    def _ensure_validation(self) -> None:
        if not self.state.validation_ran:
            self.validate()
