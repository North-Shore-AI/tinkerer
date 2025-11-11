from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import convert_scifact


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_normalize_label_handles_variants():
    assert convert_scifact.normalize_label("supports") == "supports"
    assert convert_scifact.normalize_label("SUPPORT") == "supports"
    assert convert_scifact.normalize_label("refutes") == "refutes"
    assert convert_scifact.normalize_label("Contradicted") == "refutes"


def test_convert_scifact_cli_generates_expected_completion(tmp_path: Path, scifact_claims_file: Path, scifact_corpus_file: Path):
    out_path = tmp_path / "converted.jsonl"
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "convert_scifact.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--claims_json",
        str(scifact_claims_file),
        "--corpus_json",
        str(scifact_corpus_file),
        "--out",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

    rows = read_jsonl(out_path)
    assert len(rows) == 2

    first = rows[0]
    assert "prompt" in first and "completion" in first
    completion_lines = first["completion"].splitlines()
    assert completion_lines[0] == "CLAIM[c1]: Vitamin C supplementation reduces cold duration."
    assert completion_lines[1].startswith("CLAIM[c2]:")
    assert "RELATION: c2 supports c1" in completion_lines[2]

    metadata = first["metadata"]
    assert metadata["source"] == "SciFact"
    assert metadata["claim_id"] == 1001
    assert "doc_ids" in metadata
