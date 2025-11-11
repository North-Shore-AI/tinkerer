from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def scifact_claims_file(fixtures_dir: Path) -> Path:
    return fixtures_dir / "scifact_claims_sample.jsonl"


@pytest.fixture(scope="session")
def scifact_corpus_file(fixtures_dir: Path) -> Path:
    return fixtures_dir / "scifact_corpus_sample.jsonl"


# Ensure scripts package is importable during tests.
repo_scripts_root = Path(__file__).resolve().parents[1]
if str(repo_scripts_root) not in sys.path:
    sys.path.insert(0, str(repo_scripts_root))


@pytest.fixture(scope="session")
def scifact_dataset_file(
    tmp_path_factory: pytest.TempPathFactory,
    scifact_claims_file: Path,
    scifact_corpus_file: Path,
) -> Path:
    """Run the SciFact converter once to generate a dataset for downstream tests."""
    out_dir = tmp_path_factory.mktemp("scifact_dataset")
    dataset_path = out_dir / "dataset.jsonl"
    script_path = repo_scripts_root / "scripts" / "convert_scifact.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--claims_json",
        str(scifact_claims_file),
        "--corpus_json",
        str(scifact_corpus_file),
        "--out",
        str(dataset_path),
    ]
    subprocess.run(cmd, check=True)
    return dataset_path
