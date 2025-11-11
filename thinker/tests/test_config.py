from thinker.config import load_pipeline_config


def test_load_pipeline_config(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}", encoding="utf-8")
    training_config = tmp_path / "lora.yaml"
    training_config.write_text("model: {name: test-model}", encoding="utf-8")

    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        """
tests:
  path: tests
  enabled: false

data_validation:
  path: data.jsonl
  schema:
    - name: prompt
      type: string
    - name: completion
      type: string

training:
  config_path: lora.yaml
  backend: hf_peft

evaluation:
  base_model: some-model
  checkpoint_dir: checkpoints
  claims_file: claims.jsonl
  corpus_file: corpus.jsonl
""",
        encoding="utf-8",
    )

    config = load_pipeline_config(config_path)

    assert config.data_validation is not None
    assert config.data_validation.path == dataset.resolve()
    assert config.training is not None
    assert config.training.config_path == training_config.resolve()
    assert config.evaluation is not None
    assert config.evaluation.checkpoint_dir == (tmp_path / "checkpoints").resolve()
