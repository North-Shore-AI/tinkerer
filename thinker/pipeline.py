"""High-level orchestration for Thinker workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import PipelineConfig
from .evaluation import Evaluator
from .training import create_training_backend
from .validation import DatasetValidator, TestSuiteRunner


@dataclass
class PipelineState:
    validation_ran: bool = False
    training_completed: bool = False


class ThinkerPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.state = PipelineState()

    def validate(self) -> None:
        tests_cfg = self.config.tests
        if tests_cfg and tests_cfg.enabled:
            TestSuiteRunner(tests_cfg).run()

        validation_cfg = self.config.data_validation
        if validation_cfg and validation_cfg.enabled:
            result = DatasetValidator(validation_cfg).validate()
            if not result.is_valid:
                errors = "\n".join(result.errors[:10])
                raise ValueError(f"Dataset validation failed for {result.path}:\n{errors}")

        self.state.validation_ran = True

    def train(self, backend: Optional[str] = None, skip_validation: bool = False) -> None:
        if not skip_validation:
            self._ensure_validation()
        train_cfg = self.config.training
        if not train_cfg or not train_cfg.enabled:
            return

        backend_name = backend or train_cfg.backend
        backend_cfg = train_cfg
        backend_cfg = type(train_cfg)(
            config_path=train_cfg.config_path,
            backend=backend_name,
            enabled=train_cfg.enabled,
        )
        trainer = create_training_backend(backend_cfg)
        trainer.train()
        self.state.training_completed = True

    def evaluate(self, skip_validation: bool = False) -> dict:
        if not skip_validation:
            self._ensure_validation()
        eval_cfg = self.config.evaluation
        if not eval_cfg or not eval_cfg.enabled:
            return {}
        evaluator = Evaluator(eval_cfg)
        return evaluator.run()

    def run(self, backend: Optional[str] = None) -> dict:
        self.validate()
        self.train(backend=backend, skip_validation=True)
        return self.evaluate(skip_validation=True)

    def _ensure_validation(self) -> None:
        if not self.state.validation_ran:
            self.validate()
