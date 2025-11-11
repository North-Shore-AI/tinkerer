"""Test suite runner and dataset validator utilities."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pytest

from .config import DatasetValidationConfig, SchemaField, TestSuiteConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_SCRIPT = REPO_ROOT / "cns-support-models" / "scripts" / "validate_dataset.py"


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

        if self.config.evidence_mode in ("exact", "embedding"):
            errors.extend(self._run_external_validator())

        return DatasetValidationResult(path=path, total_examples=total, errors=errors)

    def _run_external_validator(self) -> List[str]:
        if not VALIDATOR_SCRIPT.exists():
            return ["validate_dataset.py not found; cannot run external validation"]

        if not self.config.claims_path:
            return ["claims_json path required for external validation"]

        cmd = [
            sys.executable,
            str(VALIDATOR_SCRIPT),
            str(self.config.path),
            "--claims-json",
            str(self.config.claims_path),
        ]
        if self.config.max_examples:
            cmd.extend(["--max-examples", str(self.config.max_examples)])

        if self.config.evidence_mode == "embedding":
            if not self.config.corpus_path:
                return ["corpus_json path required for embedding validation"]
            cmd.extend(
                [
                    "--corpus-json",
                    str(self.config.corpus_path),
                    "--evidence-mode",
                    "embedding",
                    "--embedding-model",
                    self.config.embedding_model,
                    "--similarity-threshold",
                    str(self.config.similarity_threshold),
                ]
            )
        else:
            cmd.extend(["--evidence-mode", "exact"])

        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip() or "external validator failed"
            return [message]
        return []

    def _validate_payload(self, payload: dict, line_no: int) -> List[str]:
        issues: List[str] = []
        for field in self.config.schema:
            ok, message = field.validate(payload)
            if not ok and message:
                issues.append(f"line {line_no}: {message}")
        return issues
