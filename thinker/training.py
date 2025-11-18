"""Training backends for Thinker."""

from __future__ import annotations

import json
import subprocess
import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

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

    # Preserve prompt and completion for citation validation
    tokenized["prompt"] = prompts
    tokenized["completion"] = completions

    return tokenized


def _prepare_datasets(train_dataset, eval_dataset, tokenizer, config):
    # Keep prompt and completion columns for citation validation
    # Remove other columns that are not needed
    columns_to_remove = [
        col for col in train_dataset.column_names
        if col not in ["prompt", "completion"]
    ]

    train_dataset = train_dataset.map(
        lambda x: _tokenize_function(x, tokenizer, config),
        batched=True,
        remove_columns=columns_to_remove,
    )
    if eval_dataset:
        eval_columns_to_remove = [
            col for col in eval_dataset.column_names
            if col not in ["prompt", "completion"]
        ]
        eval_dataset = eval_dataset.map(
            lambda x: _tokenize_function(x, tokenizer, config),
            batched=True,
            remove_columns=eval_columns_to_remove,
        )

    # Set format to torch, but keep prompt and completion as strings
    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
        output_all_columns=True,
    )
    if eval_dataset:
        eval_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
            output_all_columns=True,
        )
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


class CitationAwareDataCollator:
    """Custom data collator that validates citations during training.

    This collator wraps the standard DataCollatorForLanguageModeling and adds
    citation validation to each batch. If citation validation is enabled in the
    config, it computes penalties for hallucinated citations that can be used
    in the training loss.

    Args:
        tokenizer: The tokenizer to use for padding
        config: Training configuration dict containing citation validation settings
        mlm: Whether to use masked language modeling (default: False for causal LM)
    """

    def __init__(self, tokenizer, config: Dict[str, Any], mlm: bool = False):
        from transformers import DataCollatorForLanguageModeling

        self.base_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=mlm,
        )
        self.tokenizer = tokenizer
        self.citation_weight = config.get("training", {}).get(
            "citation_validity_weight", 2.0
        )
        self.validate_citations = config.get("training", {}).get(
            "validate_citations_during_training", False
        )

    def __call__(self, features):
        """Collate features and optionally validate citations.

        Args:
            features: List of feature dicts from the dataset

        Returns:
            Batch dict with input_ids, attention_mask, labels, and optionally citation_penalties
        """
        # Use base collator for standard tokenization
        batch = self.base_collator(features)

        # Add citation validation if enabled
        if self.validate_citations:
            # Try to get prompts and completions from features
            # Note: This requires the features to contain these fields
            if features and "prompt" in features[0] and "completion" in features[0]:
                prompts = [f.get("prompt", "") for f in features]
                completions = [f.get("completion", "") for f in features]

                from .citation_validation import (
                    batch_validate_citations,
                    compute_citation_penalty,
                )

                results = batch_validate_citations(prompts, completions)

                # Store validation results for custom loss computation
                batch["citation_penalties"] = [
                    compute_citation_penalty(r, self.citation_weight)
                    for r in results
                ]

        return batch


class CitationAwareTrainer:
    """Custom Trainer that adds citation validation penalty to the training loss.

    This trainer extends the standard HuggingFace Trainer to incorporate citation
    validation into the loss function. When citation_penalties are present in the
    batch (added by CitationAwareDataCollator), they are added to the standard
    cross-entropy loss.

    This encourages the model to only cite documents that actually exist in the
    source corpus, addressing the citation hallucination problem identified in
    the HIGH severity review (claims 133, 179).
    """

    def __init__(self, *args, **kwargs):
        from transformers import Trainer

        # Create base trainer
        self.trainer = Trainer(*args, **kwargs)

        # Copy over key attributes for compatibility
        self.model = self.trainer.model
        self.args = self.trainer.args
        self.train_dataset = self.trainer.train_dataset
        self.eval_dataset = self.trainer.eval_dataset

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with optional citation penalty.

        Args:
            model: The model being trained
            inputs: Batch inputs dict
            return_outputs: Whether to return model outputs along with loss

        Returns:
            Loss tensor, or (loss, outputs) tuple if return_outputs=True
        """
        import torch

        # Standard loss computation
        outputs = model(**{k: v for k, v in inputs.items() if k != "citation_penalties"})
        loss = outputs.loss

        # Add citation penalty if available
        if "citation_penalties" in inputs and inputs["citation_penalties"]:
            penalties = torch.tensor(
                inputs["citation_penalties"],
                device=loss.device,
                dtype=loss.dtype,
            )
            citation_loss = penalties.mean()

            # Add to total loss
            loss = loss + citation_loss

            # Log citation metrics if trainer has logging
            if hasattr(self.trainer, "log"):
                self.trainer.log({
                    "citation_loss": citation_loss.item(),
                    "base_loss": (loss - citation_loss).item(),
                })

        return (loss, outputs) if return_outputs else loss

    def train(self, *args, **kwargs):
        """Forward train() call to base trainer."""
        # Replace compute_loss in base trainer
        original_compute_loss = self.trainer.compute_loss
        self.trainer.compute_loss = self.compute_loss

        try:
            return self.trainer.train(*args, **kwargs)
        finally:
            # Restore original compute_loss
            self.trainer.compute_loss = original_compute_loss

    def save_model(self, *args, **kwargs):
        """Forward save_model() call to base trainer."""
        return self.trainer.save_model(*args, **kwargs)


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

        training_args = _build_training_arguments(config, eval_dataset is not None)

        # Check if citation validation is enabled
        validate_citations = config.get("training", {}).get(
            "validate_citations_during_training", False
        )

        # Use custom collator if citation validation is enabled
        data_collator = None
        if validate_citations:
            data_collator = CitationAwareDataCollator(tokenizer, config)

        # Use CitationAwareTrainer if citation validation is enabled
        if validate_citations:
            trainer = CitationAwareTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
        else:
            from transformers import Trainer

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
