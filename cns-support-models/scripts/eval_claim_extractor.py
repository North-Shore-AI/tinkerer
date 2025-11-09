#!/usr/bin/env python3
"""
Sample from a trained claim-extractor adapter via the Tinker API.

Usage:
    python scripts/eval_claim_extractor.py \
        --prompt-file data/samples/eval_prompt.txt \
        --adapter-name claim-extractor-scifact \
        --base-model meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tinker
from tinker import types

from claim_schema import enforce_c1, parse_claim_lines, render_claim_lines


def load_prompt(path: Path | None) -> str:
    if not path:
        return (
            'Passage: "Vaccination reduces measles incidence in populations with >90% coverage."\n\n'
            "Task: Extract atomic claims and relations.\n\n"
        )
    with path.open("r", encoding="utf-8") as fh:
        return fh.read()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from the claim extractor adapter.")
    parser.add_argument("--adapter-name", default="claim-extractor-scifact")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--prompt-file", type=Path, help="Optional file containing the prompt.")
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", type=Path, help="Optional path to save prompt/completion JSON.")
    parser.add_argument("--force-c1-text", type=str, help="If set, overwrite CLAIM[c1] with this verbatim text.")
    parser.add_argument(
        "--force-c1-file",
        type=Path,
        help="Optional file containing the canonical CLAIM[c1] text (overrides --force-c1-text).",
    )
    args = parser.parse_args()

    prompt = load_prompt(args.prompt_file)
    prompt = (
        prompt.strip()
        + "\n\nOutput format:\n"
        + "CLAIM[c#]: <text>\nRELATION: <source_id> <supports|refutes|contrasts> <target_id>\n\n"
    )

    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(base_model=args.base_model)
    sampling_client = training_client.save_weights_and_get_sampling_client(name=args.adapter_name)
    tokenizer = training_client.get_tokenizer()

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    future = sampling_client.sample(
        prompt=types.ModelInput.from_ints(prompt_tokens),
        sampling_params=types.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stop=["\n\n\n"],
        ),
        num_samples=1,
    )
    result = future.result()
    decoded = tokenizer.decode(result.sequences[0].tokens)
    lines = [line.strip() for line in decoded.splitlines() if line.strip()]
    claim_lines = [line for line in lines if line.startswith("CLAIM[")]
    relation_lines = [line for line in lines if line.startswith("RELATION")]

    force_text = None
    if args.force_c1_file and args.force_c1_file.exists():
        force_text = args.force_c1_file.read_text(encoding="utf-8").strip()
    elif args.force_c1_text:
        force_text = args.force_c1_text.strip()

    if force_text:
        structured = parse_claim_lines(claim_lines)
        structured = enforce_c1(structured, force_text)
        claim_lines = render_claim_lines(structured)
        decoded = "\n".join(claim_lines + relation_lines)
    print("=== Prompt ===")
    print(prompt)
    print("=== Completion ===")
    print(decoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump({"prompt": prompt, "completion": decoded}, fh, indent=2)


if __name__ == "__main__":
    main()
