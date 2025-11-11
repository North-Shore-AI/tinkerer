from thinker.config import DatasetValidationConfig, SchemaField
from thinker.validation import DatasetValidator


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
