"""Evaluation harness for SciFact-style claim extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

from .claim_schema import parse_claim_lines
from .config import EvaluationConfig

if TYPE_CHECKING:
    from .pipeline import PipelineState

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
    def __init__(self, config: EvaluationConfig, state: Optional["PipelineState"] = None):
        self.config = config
        self.state = state

    def run(self) -> Dict[str, float]:
        completion_fn = self._build_completion_provider()
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
            completion = completion_fn(prompt)
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
        print(
            f"[eval] wrote {len(results)} completions to {self.config.output_path}",
            flush=True,
        )

        if metrics["evidence_semantic_match"]:
            metrics["evidence_semantic_match_avg"] = sum(metrics["evidence_semantic_match"]) / len(
                metrics["evidence_semantic_match"]
            )
        else:
            metrics["evidence_semantic_match_avg"] = 0.0
        metrics["c1_exact_match_rate"] = (
            metrics["c1_exact_match"] / metrics["total"] if metrics["total"] else 0.0
        )
        print(
            "[eval] metrics: "
            f"c1_exact_match_rate={metrics['c1_exact_match_rate']:.3f}, "
            f"evidence_semantic_match_avg={metrics['evidence_semantic_match_avg']:.3f}, "
            f"examples={metrics['total']}",
            flush=True,
        )
        return metrics

    def _build_completion_provider(self) -> Callable[[str], str]:
        if self.config.backend == "hf_peft":
            model, tokenizer = self._load_hf_model()
            print(
                f"[eval] using HF/PEFT checkpoint at {self.config.checkpoint_dir} (base={self.config.base_model})",
                flush=True,
            )

            def _complete(prompt: str) -> str:
                return self._generate_hf_completion(model, tokenizer, prompt)

            return _complete

        if self.config.backend == "tinker":
            adapter_info = self._resolve_tinker_adapter_info()
            print(
                "[eval] using Tinker adapter "
                f"{adapter_info['adapter_name'] or '(unnamed)'} "
                f"@ {adapter_info['adapter_path']} (base={adapter_info.get('base_model')})",
                flush=True,
            )
            sampling_client, tokenizer, types_mod = self._load_tinker_sampler(adapter_info)

            def _complete(prompt: str) -> str:
                return self._generate_tinker_completion(sampling_client, tokenizer, types_mod, prompt)

            return _complete

        raise ValueError(f"Unsupported evaluation backend '{self.config.backend}'")

    def _load_hf_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch

        if not self.config.base_model:
            raise RuntimeError("evaluation.base_model must be set for hf_peft backend")
        if not self.config.checkpoint_dir:
            raise RuntimeError("evaluation.checkpoint_dir must be set for hf_peft backend")

        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, Path(self.config.checkpoint_dir))
        tokenizer = AutoTokenizer.from_pretrained(Path(self.config.checkpoint_dir))
        model.eval()
        return model, tokenizer

    def _generate_hf_completion(self, model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
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

    def _load_tinker_sampler(self, adapter_info: Dict[str, str]):
        import tinker
        from tinker import types

        base_model = adapter_info.get("base_model") or self.config.base_model
        if not base_model:
            raise RuntimeError("Tinker evaluation requires base_model (set evaluation.base_model).")

        service_client = tinker.ServiceClient()
        sampling_client = service_client.create_sampling_client(model_path=adapter_info["adapter_path"])
        training_client = service_client.create_lora_training_client(base_model=base_model)
        tokenizer = training_client.get_tokenizer()
        return sampling_client, tokenizer, types

    def _generate_tinker_completion(self, sampling_client, tokenizer, types_mod, prompt: str) -> str:
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        future = sampling_client.sample(
            prompt=types_mod.ModelInput.from_ints(prompt_tokens),
            sampling_params=types_mod.SamplingParams(
                max_tokens=self.config.tinker_max_tokens,
                temperature=self.config.tinker_temperature,
                stop=["\n\n\n"],
            ),
            num_samples=1,
        )
        result = future.result()
        return tokenizer.decode(result.sequences[0].tokens).strip()

    def _resolve_tinker_adapter_info(self) -> Dict[str, str]:
        # Priority: explicit config, in-memory state, manifest file.
        if self.config.tinker_adapter_name and self.config.tinker_adapter_path:
            return {
                "adapter_name": self.config.tinker_adapter_name,
                "adapter_path": self.config.tinker_adapter_path,
                "base_model": self.config.base_model,
            }

        if self.state and self.state.tinker_adapter_path:
            return {
                "adapter_name": self.state.tinker_adapter_name or "",
                "adapter_path": self.state.tinker_adapter_path,
                "base_model": self.state.tinker_base_model or self.config.base_model,
            }

        manifest_path: Optional[Path] = self.config.tinker_manifest_path
        if manifest_path is None and self.state and self.state.tinker_adapter_manifest:
            manifest_path = self.state.tinker_adapter_manifest

        if manifest_path and manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            adapter_path = manifest.get("adapter_path")
            if not adapter_path:
                raise RuntimeError(f"Manifest {manifest_path} missing 'adapter_path'")
            return {
                "adapter_name": manifest.get("adapter_name") or "",
                "adapter_path": adapter_path,
                "base_model": manifest.get("base_model") or self.config.base_model,
            }

        raise RuntimeError(
            "No Tinker adapter information available. Ensure a training run has completed or set "
            "`evaluation.tinker_adapter_path` and `evaluation.tinker_adapter_name`."
        )

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
