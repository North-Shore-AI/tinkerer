from pathlib import Path

from thinker.config import load_pipeline_config
from thinker.pipeline import ThinkerPipeline
from thinker.validation import DatasetValidationResult


def _write_training_config(path: Path) -> None:
    path.write_text(
        "model:\n  name: dummy\nlora:\n  r: 1\n  lora_alpha: 1\n  lora_dropout: 0.0\n  target_modules: []\ntraining:\n  per_device_batch_size: 1\n  gradient_accumulation_steps: 1\n  learning_rate: 1e-4\n  num_epochs: 1\n  warmup_steps: 0\n  max_seq_length: 32\n  optim: adamw_torch\n  weight_decay: 0.0\n  lr_scheduler_type: cosine\ndata:\n  train_file: data.jsonl\noutput:\n  checkpoint_dir: ckpt\n  logging_steps: 1\n  eval_steps: 1\n  save_steps: 1\n",
        encoding="utf-8",
    )


def test_pipeline_runs_validation_before_training(tmp_path, monkeypatch):
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    dataset_path = fixtures_dir / "sample_data.jsonl"
    training_cfg = tmp_path / "lora.yaml"
    _write_training_config(training_cfg)

    pipeline_cfg = tmp_path / "pipeline.yaml"
    pipeline_cfg.write_text(
        f"""
tests:
  path: tests
  enabled: true

data_validation:
  path: {dataset_path}
  enabled: true
  schema:
    - name: prompt
      type: string
    - name: completion
      type: string

training:
  config_path: {training_cfg}
  backend: hf_peft

evaluation:
  base_model: dummy
  checkpoint_dir: ckpt
  claims_file: claims.jsonl
  corpus_file: corpus.jsonl
  enabled: false
""",
        encoding="utf-8",
    )

    events = []

    class DummyTestRunner:
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self):
            events.append("tests")

    class DummyValidator:
        def __init__(self, cfg):
            self.cfg = cfg

        def validate(self):
            events.append("dataset")
            return DatasetValidationResult(path=Path(self.cfg.path), total_examples=0)

    class DummyTrainer:
        def train(self):
            events.append("train")

    monkeypatch.setattr("thinker.pipeline.TestSuiteRunner", DummyTestRunner)
    monkeypatch.setattr("thinker.pipeline.DatasetValidator", DummyValidator)
    monkeypatch.setattr("thinker.pipeline.create_training_backend", lambda cfg: DummyTrainer())

    config = load_pipeline_config(pipeline_cfg)
    pipeline = ThinkerPipeline(config)
    pipeline.train(backend="hf_peft", skip_validation=False)

    assert events == ["tests", "dataset", "train"]
