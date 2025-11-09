#!/usr/bin/env python3
"""
Kickstarts LoRA training for the claim & hypothesis extractor using the Tinker API.

The script:
1. Loads a YAML config (datasets, model, and optimization hyperparameters)
2. Streams JSONL training data and converts each example into Tinker Datum objects
3. Pipelines `forward_backward` and `optim_step` calls so they share the same clock cycle
4. Emits periodic logs and optional checkpoints
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import yaml

import tinker
from tinker import types


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1])
            .decode()
            .strip()
        )
    except Exception:
        return None


@dataclass
class Example:
    prompt: str
    completion: str


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_examples(path: Path, limit: int | None = None) -> List[Example]:
    examples: List[Example] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            examples.append(Example(prompt=payload["prompt"], completion=payload["completion"]))
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"No data loaded from {path}")
    return examples


def chunk(seq: Sequence[Example], size: int) -> Iterable[Sequence[Example]]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


def _scalar(value):
    """Best-effort extraction of a float from nested arrays/tuples/dicts."""
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (tuple, list)):
        for element in value:
            try:
                return _scalar(element)
            except (TypeError, ValueError):
                continue
        return 0.0
    if isinstance(value, dict):
        for key in ("data", "value", "values"):
            if key in value:
                return _scalar(value[key])
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_datum(example: Example, tokenizer) -> types.Datum:
    prompt_tokens = tokenizer.encode(example.prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(" " + example.completion, add_special_tokens=False)

    tokens = prompt_tokens + completion_tokens
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    weights = [1 if (idx + 1) >= len(prompt_tokens) else 0 for idx in range(len(target_tokens))]

    debug_limit = int(os.environ.get("CNS_DEBUG_DATUM", 0) or 0)
    if debug_limit:
        count = getattr(build_datum, "_debug_count", 0)
        if count < debug_limit:
            print(
                f"[datum-debug] prompt={len(prompt_tokens)} completion={len(completion_tokens)} "
                f"input={len(input_tokens)} target={len(target_tokens)} "
                f"weights_sum={sum(weights)} first_weights={weights[:10]}",
                flush=True,
            )
            build_datum._debug_count = count + 1

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(
            target_tokens=tinker.TensorData(
                data=[int(tok) for tok in target_tokens],
                dtype="int64",
                shape=[len(target_tokens)],
            ),
            weights=tinker.TensorData(
                data=[float(w) for w in weights],
                dtype="float32",
                shape=[len(weights)],
            ),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CNS claim extractor via Tinker.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument("--log-dir", type=Path, default=Path("runs"), help="Directory to store provenance logs.")
    args = parser.parse_args()

    config_path = args.config.resolve()
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    opt_cfg = cfg["optimization"]
    model_cfg = cfg["model"]
    log_cfg = cfg.get("logging", {})

    project_root = config_path.parent.parent
    train_path = (project_root / data_cfg["train_path"]).resolve()
    examples = load_examples(train_path, limit=data_cfg.get("max_samples"))

    print("[init] Preparing Tinker client...", flush=True)
    service_client = tinker.ServiceClient()
    print("[init] Creating LoRA training client (this may take ~1 min)...", flush=True)
    training_client = service_client.create_lora_training_client(base_model=model_cfg["base_model"])
    print("[init] Training client ready.", flush=True)
    tokenizer = training_client.get_tokenizer()

    steps_per_epoch = math.ceil(len(examples) / data_cfg["batch_size"])
    total_steps = steps_per_epoch * opt_cfg["epochs"]

    print(f"[init] Loaded {len(examples)} examples from {train_path}")
    print(f"[init] Training {model_cfg['adapter_name']} for {opt_cfg['epochs']} epochs ({total_steps} steps)")

    step = 0
    last_loss = None
    print("[init] Entering training loop...", flush=True)
    for epoch in range(opt_cfg["epochs"]):
        random.shuffle(examples)
        for batch in chunk(examples, data_cfg["batch_size"]):
            datums = [build_datum(ex, tokenizer) for ex in batch]
            if os.environ.get("CNS_DEBUG_DATUM"):
                for idx, datum in enumerate(datums[:1]):
                    weights_arr = datum.loss_fn_inputs["weights"]
                    try:
                        weights_list = weights_arr.tolist()
                    except AttributeError:
                        weights_list = list(weights_arr)
                    print(
                        f"[debug-datum] batch={step+1} example={idx} weights_len={len(weights_list)} "
                        f"sum={sum(weights_list)} first={weights_list[:10]}",
                        flush=True,
                    )
            print(f"[debug] submitting batch of {len(datums)} examples to forward_backward", flush=True)
            fwdbwd_future = training_client.forward_backward(datums, loss_fn="cross_entropy")
            print("[debug] submitting optim_step", flush=True)
            optim_future = training_client.optim_step(
                types.AdamParams(learning_rate=opt_cfg["learning_rate"])
            )

            print("[debug] waiting for forward_backward result...", flush=True)
            fwdbwd_result = fwdbwd_future.result()
            if step == 0:
                first_output = fwdbwd_result.loss_fn_outputs[0]
                print(
                    f"[debug] loss_fn_output_keys={list(first_output.keys())}",
                    flush=True,
                )
                print(
                    f"[debug] first loss_fn_output sample={first_output}",
                    flush=True,
                )
                print(
                    f"[debug] metrics={fwdbwd_result.metrics}",
                    flush=True,
                )
            print("[debug] waiting for optim_step result...", flush=True)
            optim_future.result()
            print("[debug] batch completed", flush=True)

            loss = fwdbwd_result.metrics.get("loss:mean")
            if loss is None and "loss:sum" in fwdbwd_result.metrics:
                loss = fwdbwd_result.metrics["loss:sum"] / max(
                    fwdbwd_result.metrics.get("num_tokens", 1), 1
                )
            if loss is None:
                loss = 0.0
            last_loss = loss

            step += 1
            if step % 10 == 0 or step == 1:
                print(f"[train] epoch={epoch+1} step={step}/{total_steps} loss={loss:.4f}")

            eval_every = log_cfg.get("eval_every_steps")
            if eval_every and step % eval_every == 0:
                sample_future = training_client.sample(
                    prompt=types.ModelInput.from_ints(tokenizer.encode(batch[0].prompt)),
                    sampling_params=types.SamplingParams(max_tokens=128, temperature=0.0),
                    num_samples=1,
                )
                sample_result = sample_future.result()
                decoded = tokenizer.decode(sample_result.sequences[0].tokens)
                print(f"[eval] sample output (truncated): {decoded[:120]!r}")

            save_every = log_cfg.get("save_every_steps")
            if save_every and step % save_every == 0:
                training_client.save_weights(name=model_cfg["adapter_name"])
                print(f"[checkpoint] Saved adapter snapshot at step {step}")

    final_client = training_client.save_weights_and_get_sampling_client(name=model_cfg["adapter_name"])
    print("[done] Saved adapter weights. Ready for offline evals via sampling.")

    if args.log_dir:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        now_utc = dt.datetime.now(dt.timezone.utc)
        metadata = {
            "timestamp": now_utc.isoformat().replace("+00:00", "Z"),
            "config": str(config_path),
            "git_commit": git_commit(),
            "adapter_name": model_cfg["adapter_name"],
            "base_model": model_cfg["base_model"],
            "dataset": {
                "train_path": str(train_path),
                "num_examples": len(examples),
                "sha256": sha256_file(train_path),
            },
            "optimization": {
                "epochs": opt_cfg["epochs"],
                "batch_size": data_cfg["batch_size"],
                "learning_rate": opt_cfg["learning_rate"],
            },
            "logging_notes": log_cfg.get("notes"),
            "total_steps": total_steps,
            "final_loss": last_loss,
        }
        log_path = args.log_dir / f"train_{model_cfg['adapter_name']}_{now_utc.strftime('%Y%m%dT%H%M%SZ')}.json"
        with log_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        print(f"[log] wrote provenance metadata to {log_path}")


if __name__ == "__main__":
    main()
