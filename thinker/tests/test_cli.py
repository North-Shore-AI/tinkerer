from __future__ import annotations

from unittest import mock

import pytest

from thinker import cli


def test_data_setup_cli_invokes_helper(monkeypatch):
    called = {}

    def fake_run_data_setup(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(cli, "run_data_setup", fake_run_data_setup)
    exit_code = cli.main(["data", "setup", "--skip-validation"])
    assert exit_code == 0
    assert called["dataset"] == "scifact"
    assert called["skip_validation"] is True
    assert called["validation_mode"] == "exact"
    assert str(called["clean_output"]).endswith("scifact_claim_extractor_clean.jsonl")
    assert called["filter_invalid"] is True


def test_data_setup_cli_embedding(monkeypatch):
    called = {}

    def fake_run_data_setup(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(cli, "run_data_setup", fake_run_data_setup)
    exit_code = cli.main(
        [
            "data",
            "setup",
            "--dataset",
            "fever",
            "--validation-mode",
            "embedding",
            "--similarity-threshold",
            "0.8",
        ]
    )
    assert exit_code == 0
    assert called["dataset"] == "fever"
    assert called["validation_mode"] == "embedding"
    assert called["similarity_threshold"] == 0.8
    assert str(called["clean_output"]).endswith("scifact_claim_extractor_clean.jsonl")
    assert called["filter_invalid"] is True
