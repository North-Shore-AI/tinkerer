"""Test suite runner and dataset validator utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pytest

from .config import DatasetValidationConfig, SchemaField, TestSuiteConfig


@dataclass
class DatasetValidationResult:
    path: Path
    total_examples: int
    errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


class TestSuiteRunner:
    """Runs pytest with the configuration defined in Thinker configs."""

    def __init__(self, config: TestSuiteConfig):
        self.config = config

    def run(self) -> None:
        if not self.config.enabled:
            return

        args: List[str] = [str(self.config.path)]
        if self.config.markers:
            args.extend(["-m", self.config.markers])
        args.extend(self.config.args)

        exit_code = pytest.main(args)
        if exit_code != 0:
            raise RuntimeError(f"pytest exited with code {exit_code}")


class DatasetValidator:
    """Validates JSONL payloads based on a schema definition."""

    def __init__(self, config: DatasetValidationConfig):
        self.config = config

    def validate(self) -> DatasetValidationResult:
        if not self.config.enabled:
            return DatasetValidationResult(path=self.config.path, total_examples=0)

        path = Path(self.config.path)
        errors: List[str] = []
        total = 0

        with path.open("r", encoding="utf-8") as fh:
            for idx, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                total += 1
                if self.config.max_examples and total > self.config.max_examples:
                    break
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"line {idx}: invalid JSON ({exc})")
                    continue
                errors.extend(self._validate_payload(payload, idx))

        return DatasetValidationResult(path=path, total_examples=total, errors=errors)

    def _validate_payload(self, payload: dict, line_no: int) -> List[str]:
        issues: List[str] = []
        for field in self.config.schema:
            ok, message = field.validate(payload)
            if not ok and message:
                issues.append(f"line {line_no}: {message}")
        return issues
