"""Training backends for Thinker."""

from __future__ import annotations

import json
import subprocess
import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from .config import LocalTrainingConfig


class TrainingBackend:
    """Simple protocol describing a training backend."""

    def train(self) -> TrainingReport:
        raise NotImplementedError


@dataclass
class TrainingReport:
    checkpoint_dir: Path
    metrics: Dict[str, float]
    backend: str


def create_training_backend(config: LocalTrainingConfig) -> TrainingBackend:
    if config.backend == "hf_peft":
        return LocalPEFTTrainer(config.config_path)
    if config.backend == "tinker":
        return TinkerScriptTrainer(config)
    raise ValueError(f"Unsupported training backend '{config.backend}'")


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = (base_dir / raw_path).resolve()
    return candidate


def _setup_model_and_tokenizer(config: Dict[str, Any]):
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    import torch
    from peft import prepare_model_for_kbit_training

    model_cfg = config["model"]
    model_name = model_cfg["name"]

    quantization = model_cfg.get("quantization")
    bnb_config = None
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=model_cfg.get("device_map", "auto"),
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    if bnb_config:
        model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def _apply_lora(model, config: Dict[str, Any]):
    from peft import LoraConfig, TaskType, get_peft_model

    lora_cfg = config["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    return model


def _load_datasets(config: Dict[str, Any], base_dir: Path):
    from datasets import load_dataset

    data_cfg = config["data"]
    train_file = _resolve_path(base_dir, data_cfg["train_file"])
    dataset = load_dataset("json", data_files=str(train_file), split="train")

    eval_dataset = None
    split_ratio = data_cfg.get("validation_split", 0)
    if split_ratio:
        dataset = dataset.train_test_split(
            test_size=split_ratio,
            seed=data_cfg.get("seed", 42),
        )
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        train_dataset = dataset

    return train_dataset, eval_dataset


def _tokenize_function(examples, tokenizer, config: Dict[str, Any]):
    max_length = config["training"]["max_seq_length"]

    prompts = examples["prompt"]
    completions = examples["completion"]
    combined = [f"{p}\n{c}" for p, c in zip(prompts, completions)]

    tokenized = tokenizer(
        combined,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    labels = []
    for prompt, input_ids in zip(prompts, tokenized["input_ids"]):
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        mask_length = min(len(prompt_tokens), len(input_ids))
        sample_labels = list(input_ids)
        for idx in range(mask_length):
            sample_labels[idx] = -100
        labels.append(sample_labels)
    tokenized["labels"] = labels
    return tokenized


def _prepare_datasets(train_dataset, eval_dataset, tokenizer, config):
    train_dataset = train_dataset.map(
        lambda x: _tokenize_function(x, tokenizer, config),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            lambda x: _tokenize_function(x, tokenizer, config),
            batched=True,
            remove_columns=eval_dataset.column_names,
        )

    train_dataset.set_format(type="torch")
    if eval_dataset:
        eval_dataset.set_format(type="torch")
    return train_dataset, eval_dataset


def _build_training_arguments(config: Dict[str, Any], has_eval: bool):
    from transformers import TrainingArguments

    training_cfg = config["training"]
    output_cfg = config["output"]

    eval_steps = output_cfg.get("eval_steps")
    args = TrainingArguments(
        output_dir=output_cfg["checkpoint_dir"],
        num_train_epochs=training_cfg["num_epochs"],
        per_device_train_batch_size=training_cfg["per_device_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        warmup_steps=training_cfg["warmup_steps"],
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        optim=training_cfg["optim"],
        weight_decay=training_cfg["weight_decay"],
        fp16=training_cfg.get("fp16", False),
        logging_steps=output_cfg["logging_steps"],
        save_steps=output_cfg["save_steps"],
        save_total_limit=output_cfg.get("save_total_limit"),
        eval_steps=eval_steps if has_eval else None,
        report_to=output_cfg.get("report_to", "none"),
        remove_unused_columns=False,
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", False),
        evaluation_strategy="steps" if has_eval else "no",
        load_best_model_at_end=has_eval,
        push_to_hub=False,
    )
    return args


class LocalPEFTTrainer(TrainingBackend):
    """Runs PEFT training locally via Hugging Face transformers."""

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path).resolve()

    def train(self) -> None:
        config = _load_yaml(self.config_path)
        config_dir = self.config_path.parent
        config["data"]["train_file"] = str(_resolve_path(config_dir, config["data"]["train_file"]))
        config["output"]["checkpoint_dir"] = str(
            _resolve_path(config_dir, config["output"]["checkpoint_dir"])
        )

        model, tokenizer = _setup_model_and_tokenizer(config)
        model = _apply_lora(model, config)
        train_dataset, eval_dataset = _load_datasets(config, config_dir)
        train_dataset, eval_dataset = _prepare_datasets(train_dataset, eval_dataset, tokenizer, config)

        from transformers import Trainer

        training_args = _build_training_arguments(config, eval_dataset is not None)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )

        trainer.train()

        checkpoint_dir = Path(config["output"]["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        return TrainingReport(
            checkpoint_dir=checkpoint_dir,
            metrics={},
            backend="hf_peft",
        )


class TinkerScriptTrainer(TrainingBackend):
    """Runs the legacy Tinker training script via subprocess."""

    def __init__(self, config: LocalTrainingConfig):
        self.config = config

    def train(self) -> TrainingReport:
        if not os.environ.get("TINKER_API_KEY"):
            raise RuntimeError("TINKER_API_KEY is not set; export it before running the Tinker backend.")
        script = self.config.tinker_script or (
            Path(__file__).resolve().parents[1] / "cns-support-models" / "scripts" / "train_claim_extractor.py"
        )
        script = script.resolve()
        config_path = (
            self.config.tinker_config_path or self.config.config_path
        ).resolve()
        cmd = [
            sys.executable,
            str(script),
            "--config",
            str(config_path),
        ]
        if self.config.log_dir:
            cmd.extend(["--log-dir", str(self.config.log_dir.resolve())])
        subprocess.run(cmd, cwd=script.parent, check=True)

        adapter_metrics: Dict[str, Any] = {}
        if self.config.log_dir:
            manifest_path = self.config.log_dir / "latest_tinker_adapter.json"
            if manifest_path.exists():
                try:
                    with manifest_path.open("r", encoding="utf-8") as fh:
                        manifest = json.load(fh)
                    adapter_metrics = {
                        "adapter_name": manifest.get("adapter_name"),
                        "adapter_path": manifest.get("adapter_path"),
                        "manifest_path": str(manifest_path),
                        "base_model": manifest.get("base_model"),
                    }
                except Exception as exc:  # noqa: BLE001
                    adapter_metrics["manifest_error"] = str(exc)

        return TrainingReport(
            checkpoint_dir=Path(),
            metrics=adapter_metrics,
            backend="tinker",
        )
