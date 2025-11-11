"""Evaluation harness for SciFact-style claim extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .claim_schema import parse_claim_lines
from .config import EvaluationConfig


def _load_jsonl(path: Path) -> List[Dict]:
    items: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                items.append(json.loads(line))
    return items


def _load_corpus(path: Path) -> Dict[str, Dict]:
    corpus = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                doc = json.loads(line)
                corpus[str(doc["doc_id"])] = doc
    return corpus


def evaluate_semantic_match(predicted: List[str], gold_sentences: List[str]) -> float:
    matches = 0
    for pred in predicted:
        for gold in gold_sentences:
            if pred.strip() == gold.strip():
                matches += 1
                break
    return matches / len(predicted) if predicted else 0.0


class Evaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def run(self) -> Dict[str, float]:
        model, tokenizer = self._load_model()
        claims = _load_jsonl(Path(self.config.claims_file))[: self.config.max_samples]
        corpus = _load_corpus(Path(self.config.corpus_file))

        metrics = {
            "total": len(claims),
            "c1_exact_match": 0,
            "evidence_semantic_match": [],
        }
        results = []

        for claim in claims:
            prompt = self._build_prompt(claim, corpus)
            completion = self._generate_completion(model, tokenizer, prompt)
            parsed = parse_claim_lines(line for line in completion.splitlines() if line.startswith("CLAIM["))

            c1 = parsed.get("c1")
            if c1 and c1.text == claim["claim"]:
                metrics["c1_exact_match"] += 1

            evidence_claims = [entry.text for key, entry in parsed.items() if key != "c1"]
            gold_sentences = self._gather_gold_sentences(claim, corpus)

            if evidence_claims and gold_sentences:
                metrics["evidence_semantic_match"].append(
                    evaluate_semantic_match(evidence_claims, gold_sentences)
                )

            results.append({"claim_id": claim["id"], "prompt": prompt, "completion": completion})

        self._write_results(results)

        if metrics["evidence_semantic_match"]:
            metrics["evidence_semantic_match_avg"] = sum(metrics["evidence_semantic_match"]) / len(
                metrics["evidence_semantic_match"]
            )
        else:
            metrics["evidence_semantic_match_avg"] = 0.0
        metrics["c1_exact_match_rate"] = (
            metrics["c1_exact_match"] / metrics["total"] if metrics["total"] else 0.0
        )
        return metrics

    def _load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch

        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, Path(self.config.checkpoint_dir))
        tokenizer = AutoTokenizer.from_pretrained(Path(self.config.checkpoint_dir))
        model.eval()
        return model, tokenizer

    def _generate_completion(self, model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
        import torch

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt) :].strip()

    def _build_prompt(self, claim: Dict, corpus: Dict[str, Dict]) -> str:
        prompt = f"Given the following hypothesis, extract claims and relations:\n\n"
        prompt += f"Hypothesis: {claim['claim']}\n\n"
        for doc_id in claim.get("cited_doc_ids", []):
            doc = corpus.get(str(doc_id))
            if not doc:
                continue
            prompt += f"Document {doc_id}: {doc['title']}\n"
            prompt += " ".join(doc["abstract"]) + "\n\n"
        prompt += "Extract claims and relations:"
        return prompt

    def _gather_gold_sentences(self, claim: Dict, corpus: Dict[str, Dict]) -> List[str]:
        sentences: List[str] = []
        evidence = claim.get("evidence", {})
        for doc_id, entries in evidence.items():
            doc = corpus.get(str(doc_id))
            if not doc:
                continue
            for entry in entries:
                for idx in entry.get("sentences", []):
                    if 0 <= idx < len(doc["abstract"]):
                        sentences.append(doc["abstract"][idx])
        return sentences

    def _write_results(self, results: List[Dict[str, str]]) -> None:
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            for row in results:
                fh.write(json.dumps(row) + "\n")
