from pathlib import Path

import pytest

from thinker import training
from thinker.config import LocalTrainingConfig


def test_create_tinker_backend_invokes_script(monkeypatch, tmp_path):
    script = tmp_path / "train.py"
    script.write_text("print('stub')")
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text("model:\n  base_model: stub\n")
    log_dir = tmp_path / "logs"

    called = {}

    def fake_run(cmd, cwd, check):
        called["cmd"] = cmd
        called["cwd"] = cwd
        called["check"] = check

    monkeypatch.setattr(training.subprocess, "run", fake_run)
    monkeypatch.setenv("TINKER_API_KEY", "dummy")

    cfg = LocalTrainingConfig(
        config_path=config_yaml,
        backend="tinker",
        tinker_script=script,
        log_dir=log_dir,
    )

    backend = training.create_training_backend(cfg)
    backend.train()

    assert called["cmd"][0].endswith("python") or called["cmd"][0].endswith("python3")
    assert "--config" in called["cmd"]
    assert str(config_yaml.resolve()) in called["cmd"]
    assert "--log-dir" in called["cmd"]
    assert called["cwd"] == script.parent
