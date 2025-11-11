from thinker.config import DatasetValidationConfig, SchemaField
from thinker.validation import DatasetValidator, subprocess, VALIDATOR_SCRIPT


def test_dataset_validator_passes_on_valid_payload(tmp_path):
    data_file = tmp_path / "valid.jsonl"
    data_file.write_text(
        '{"prompt": "P", "completion": "CLAIM[c1]: P"}\n',
        encoding="utf-8",
    )
    cfg = DatasetValidationConfig(
        path=data_file,
        schema=[SchemaField("prompt"), SchemaField("completion")],
    )
    result = DatasetValidator(cfg).validate()
    assert result.is_valid
    assert result.total_examples == 1


def test_dataset_validator_reports_schema_errors(tmp_path):
    data_file = tmp_path / "invalid.jsonl"
    data_file.write_text(
        '{"prompt": "P"}\n{"prompt": "", "completion": ""}\n',
        encoding="utf-8",
    )
    cfg = DatasetValidationConfig(
        path=data_file,
        schema=[SchemaField("prompt"), SchemaField("completion")],
    )
    result = DatasetValidator(cfg).validate()
    assert not result.is_valid
    assert any("missing required field 'completion'" in err for err in result.errors)
    assert any("cannot be empty" in err for err in result.errors)


def test_dataset_validator_runs_external_exact(monkeypatch, tmp_path):
    data_file = tmp_path / "valid.jsonl"
    data_file.write_text('{"prompt": "P", "completion": "CLAIM[c1]: P"}\n', encoding="utf-8")
    claims_file = tmp_path / "claims.jsonl"
    claims_file.write_text('{"id": 1, "claim": "P"}\n', encoding="utf-8")

    called = {}

    def fake_run(cmd, cwd, capture_output, text):
        called["cmd"] = cmd
        called["cwd"] = cwd

        class Result:
            def __init__(self):
                self.returncode = 0
                self.stdout = ""
                self.stderr = ""

        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    cfg = DatasetValidationConfig(
        path=data_file,
        schema=[SchemaField("prompt"), SchemaField("completion")],
        evidence_mode="exact",
        claims_path=claims_file,
        max_examples=5,
    )
    result = DatasetValidator(cfg).validate()
    assert result.is_valid
    assert "--claims-json" in called["cmd"]
    assert "--evidence-mode" in called["cmd"]


def test_dataset_validator_requires_corpus_for_embedding(tmp_path):
    data_file = tmp_path / "data.jsonl"
    data_file.write_text('{"prompt":"P","completion":"CLAIM[c1]: P"}\n', encoding="utf-8")
    claims_file = tmp_path / "claims.jsonl"
    claims_file.write_text('{"id": 1, "claim": "P"}\n', encoding="utf-8")

    cfg = DatasetValidationConfig(
        path=data_file,
        schema=[SchemaField("prompt"), SchemaField("completion")],
        evidence_mode="embedding",
        claims_path=claims_file,
        corpus_path=None,
    )
    result = DatasetValidator(cfg).validate()
    assert not result.is_valid
    assert any("corpus_json path required" in err for err in result.errors)
