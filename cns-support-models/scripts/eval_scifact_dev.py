#!/usr/bin/env python3
"""
Lightweight structured evaluation against SciFact dev claims.

For each claim, we build the same prompt used during training, sample from the
trained adapter, parse the CLAIM/RELATION outputs, and compute simple metrics:
    - Did any predicted claim contain the gold claim text?
    - How many claims / relations were produced?

Usage:
    python scripts/eval_scifact_dev.py \
        --config configs/claim_extractor_scifact.yaml \
        --max-samples 50 \
        --output runs/scifact_dev_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

import tinker
from tinker import types


def load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_corpus(path: Path) -> Dict[int, dict]:
    corpus = {}
    for entry in load_jsonl(path):
        corpus[entry["doc_id"]] = entry
    return corpus


def extract_passage(claim_entry: dict, corpus: Dict[int, dict]) -> str:
    docs = []
    for doc_id in claim_entry.get("cited_doc_ids", []):
        doc = corpus.get(doc_id)
        if not doc:
            continue
        title = doc.get("title", "")
        abstract = " ".join(doc.get("abstract", []))
        docs.append(f"{title}. {abstract}".strip())
    return "\n\n".join(docs) if docs else ""


def parse_completion(text: str) -> Tuple[List[str], List[str]]:
    claims = []
    relations = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("CLAIM["):
            claims.append(line)
        elif line.startswith("RELATION:"):
            relations.append(line)
    raw_claims = []
    raw_relations = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("CLAIM["):
            raw_claims.append(stripped)
        elif stripped.startswith("RELATION:"):
            raw_relations.append(stripped)
    if not raw_claims:
        # Attempt to coerce common "Claim 1:" style outputs
        normalized = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("claim") and ":" in stripped:
                prefix, rest = stripped.split(":", 1)
                normalized.append(f"CLAIM[c{len(normalized)+1}]: {rest.strip()}")
        raw_claims = normalized
    return raw_claims, raw_relations


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate adapter on SciFact dev set.")
    parser.add_argument("--config", type=Path, required=True, help="Training config YAML")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--output", type=Path, help="Optional JSONL to store predictions")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    project_root = args.config.parent.parent
    corpus = load_corpus(project_root / "data/raw/scifact/corpus.jsonl")
    dev_claims = load_jsonl(project_root / "data/raw/scifact/claims_dev.jsonl")
    dev_claims = [c for c in dev_claims if c.get("cited_doc_ids")]
    if args.max_samples:
        dev_claims = dev_claims[: args.max_samples]

    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        print("[error] TINKER_API_KEY is not set; aborting evaluation.")
        return
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=cfg["model"]["base_model"]
    )
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=cfg["model"]["adapter_name"]
    )
    tokenizer = training_client.get_tokenizer()

    results = []
    matched = 0
    for idx, claim_entry in enumerate(dev_claims, start=1):
        passage = extract_passage(claim_entry, corpus)
        if not passage:
            continue
        prompt = (
            f"Passage:\n{passage}\n\n"
            "Task: Extract atomic claims and relations.\n\n"
            "Output format:\nCLAIM[c#]: <text>\nRELATION: <src_id> <supports|refutes> <dst_id>\n\n"
        )
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        future = sampling_client.sample(
            prompt=types.ModelInput.from_ints(prompt_tokens),
            sampling_params=types.SamplingParams(
                max_tokens=160,
                temperature=args.temperature,
                stop=["\n\n\n"],
            ),
            num_samples=1,
        )
        completion = tokenizer.decode(future.result().sequences[0].tokens)
        claims_pred, relations_pred = parse_completion(completion)
        gold_claim = claim_entry["claim"].lower()
        if any(gold_claim in c.lower() for c in claims_pred):
            matched += 1
        results.append(
            {
                "id": claim_entry["id"],
                "gold_claim": claim_entry["claim"],
                "predicted": completion,
                "num_claims": len(claims_pred),
                "num_relations": len(relations_pred),
            }
        )
        if idx % 10 == 0:
            print(f"[eval] processed {idx} samples")
            print(f"[eval] sample prediction: {completion[:200]!r}")

    accuracy = matched / len(results) if results else 0.0
    print(f"[summary] samples={len(results)} claim_match_rate={accuracy:.2%}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            for item in results:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[summary] wrote detailed predictions to {args.output}")


if __name__ == "__main__":
    main()
