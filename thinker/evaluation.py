"""Evaluation harness for SciFact-style claim extraction.

Updated to use 4-stage semantic validation per AGENTS.md Section 4.1.
Exact-match metrics are retained for legacy comparison only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import pstdev
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .claim_schema import parse_claim_lines, parse_relation_line
from .config import EvaluationConfig
from .logic import compute_graph_stats
from .metrics import ChiralityAnalyzer, build_fisher_rao_stats
from .semantic_validation import SemanticValidator

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


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _moving_average(values: List[float], window: int) -> List[float]:
    if not values:
        return []
    result: List[float] = []
    for idx in range(len(values)):
        lo = max(0, idx - window + 1)
        window_values = [v for v in values[lo : idx + 1] if v is not None]
        if not window_values:
            result.append(None)
        else:
            result.append(sum(window_values) / len(window_values))
    return result


def build_evaluation_series(samples: List[Dict[str, Any]], moving_avg_window: int = 5) -> Dict[str, Any]:
    """Create cumulative + moving-average series for dashboard charting."""

    if not samples:
        return {
            "indices": [],
            "timestamps": [],
            "cumulative_rates": {},
            "value_series": {},
            "moving_averages": {},
        }

    indices = [sample.get("index", idx + 1) for idx, sample in enumerate(samples)]
    timestamps = [sample.get("timestamp") for sample in samples]

    cumulative_keys = {
        "schema_valid": "schema_valid",
        "citation_valid": "citation_valid",
        "entailment_pass": "entailment_pass",
        "similarity_pass": "similarity_pass",
        "paraphrase_accepted": "paraphrase_accepted",
        "overall_pass": "overall_pass",
    }
    cumulative_counts = {key: 0 for key in cumulative_keys}
    cumulative_rates: Dict[str, List[float]] = {key: [] for key in cumulative_keys}

    numeric_series = {
        "entailment_score": [],
        "semantic_similarity": [],
        "beta1": [],
        "chirality_score": [],
        "fisher_rao_distance": [],
    }
    binary_series = {
        "citation_valid": [],
        "schema_valid": [],
        "overall_pass": [],
    }

    for idx, sample in enumerate(samples, start=1):
        for key, sample_key in cumulative_keys.items():
            value = sample.get(sample_key)
            cumulative_counts[key] += 1 if value else 0
            cumulative_rates[key].append(cumulative_counts[key] / idx)

        numeric_series["entailment_score"].append(sample.get("entailment_score"))
        numeric_series["semantic_similarity"].append(sample.get("semantic_similarity"))
        numeric_series["beta1"].append(sample.get("beta1"))
        numeric_series["chirality_score"].append(sample.get("chirality_score"))
        numeric_series["fisher_rao_distance"].append(sample.get("fisher_rao_distance"))

        binary_series["citation_valid"].append(1 if sample.get("citation_valid") else 0)
        binary_series["schema_valid"].append(1 if sample.get("schema_valid") else 0)
        binary_series["overall_pass"].append(1 if sample.get("overall_pass") else 0)

    moving_keys = {
        "entailment_score": numeric_series["entailment_score"],
        "semantic_similarity": numeric_series["semantic_similarity"],
        "chirality_score": numeric_series["chirality_score"],
    }
    moving_averages = {
        key: _moving_average(values, moving_avg_window)
        for key, values in moving_keys.items()
    }

    return {
        "indices": indices,
        "timestamps": timestamps,
        "cumulative_rates": cumulative_rates,
        "value_series": {**numeric_series, **binary_series},
        "moving_averages": moving_averages,
    }


class Evaluator:
    def __init__(self, config: EvaluationConfig, state: Optional["PipelineState"] = None):
        self.config = config
        self.state = state
        # Initialize semantic validator for 4-stage validation
        self.semantic_validator = None  # Lazy init in run() to avoid loading models unnecessarily

    def run(self) -> Dict[str, float]:
        # Initialize semantic validator (lazy load)
        if self.semantic_validator is None:
            self.semantic_validator = SemanticValidator()

        completion_fn = self._build_completion_provider()
        claims = _load_jsonl(Path(self.config.claims_file))[: self.config.max_samples]
        corpus = _load_corpus(Path(self.config.corpus_file))

        chirality_analyzer: Optional[ChiralityAnalyzer] = None
        if claims:
            gold_texts = [entry["claim"] for entry in claims]
            gold_vectors = self.semantic_validator.embedding_model.encode(
                gold_texts, convert_to_numpy=True
            )
            fr_stats = build_fisher_rao_stats(gold_vectors)
            chirality_analyzer = ChiralityAnalyzer(self.semantic_validator.embedding_model, fr_stats)

        # Initialize metrics tracking all 4 validation stages
        metrics = {
            "total": len(claims),
            # Schema compliance (prerequisite check)
            "schema_valid": 0,
            # Stage 1: Citation Accuracy
            "citation_valid": 0,
            # Stage 2: Entailment
            "entailment_scores": [],
            "entailment_pass": 0,
            # Stage 3: Semantic Similarity
            "similarity_scores": [],
            "similarity_pass": 0,
            # Stage 4: Paraphrase Tolerance
            "paraphrase_accepted": 0,
            # Overall pass rate
            "overall_pass": 0,
            # Graph / topology metrics
            "beta1_values": [],
            "beta1_nonzero": 0,
            # Chirality metrics
            "chirality_scores": [],
            "fisher_rao_distances": [],
            # Legacy exact-match metrics (for comparison only)
            "c1_exact_match": 0,
            "evidence_exact_match": [],
        }
        results = []
        sample_metrics: List[Dict[str, Any]] = []

        total_samples = len(claims)
        for idx, claim in enumerate(claims, start=1):
            print(f"[eval] sample {idx}/{total_samples} claim_id={claim.get('id')}", flush=True)
            prompt = self._build_prompt(claim, corpus)
            completion = completion_fn(prompt)

            # Parse claims from output
            claim_lines = [line for line in completion.splitlines() if line.strip().upper().startswith("CLAIM[")]
            parsed = parse_claim_lines(claim_lines)
            c1 = parsed.get("c1")
            generated_claim_text = c1.text if c1 else ""

            # Get gold evidence IDs
            gold_evidence_ids = set(str(doc_id) for doc_id in claim.get("cited_doc_ids", []))

            # Run 4-stage semantic validation
            validation = self.semantic_validator.validate_claim(
                generated_claim=generated_claim_text,
                gold_claim=claim["claim"],
                generated_full_output=completion,
                evidence_corpus=corpus,
                gold_evidence_ids=gold_evidence_ids,
            )

            # Graph topology
            relations = [
                relation
                for relation in (parse_relation_line(line) for line in completion.splitlines())
                if relation is not None
            ]
            graph_stats = compute_graph_stats(parsed.keys(), relations)
            metrics["beta1_values"].append(graph_stats.beta1)
            if graph_stats.beta1 > 0:
                metrics["beta1_nonzero"] += 1

            pred_evidence_ids = self.semantic_validator._extract_citation_ids(completion)
            evidence_overlap = (
                len(pred_evidence_ids & gold_evidence_ids) / max(len(gold_evidence_ids) or 1, 1)
                if gold_evidence_ids
                else 0.0
            )
            chirality = None
            if chirality_analyzer:
                chirality = chirality_analyzer.compare(
                    generated_claim_text,
                    claim["claim"],
                    evidence_overlap=evidence_overlap,
                    polarity_conflict=graph_stats.polarity_conflict,
                )
                metrics["chirality_scores"].append(chirality.chirality_score)
                metrics["fisher_rao_distances"].append(chirality.fisher_rao_distance)

            # Track metrics from validation
            if validation.schema_valid:
                metrics["schema_valid"] += 1
            if validation.citation_valid:
                metrics["citation_valid"] += 1

            metrics["entailment_scores"].append(validation.entailment_score)
            if validation.entailment_pass:
                metrics["entailment_pass"] += 1

            metrics["similarity_scores"].append(validation.semantic_similarity)
            if validation.similarity_pass:
                metrics["similarity_pass"] += 1

            if validation.paraphrase_accepted:
                metrics["paraphrase_accepted"] += 1

            if validation.overall_pass:
                metrics["overall_pass"] += 1

            # Legacy exact-match metrics (for comparison)
            if c1 and c1.text == claim["claim"]:
                metrics["c1_exact_match"] += 1

            evidence_claims = [entry.text for key, entry in parsed.items() if key != "c1"]
            gold_sentences = self._gather_gold_sentences(claim, corpus)
            if evidence_claims and gold_sentences:
                metrics["evidence_exact_match"].append(
                    evaluate_semantic_match(evidence_claims, gold_sentences)
                )

            record = {
                "claim_id": claim["id"],
                "prompt": prompt,
                "completion": completion,
                "validation": {
                    "schema_valid": validation.schema_valid,
                    "citation_valid": validation.citation_valid,
                    "entailment_score": validation.entailment_score,
                    "entailment_pass": validation.entailment_pass,
                    "semantic_similarity": validation.semantic_similarity,
                    "similarity_pass": validation.similarity_pass,
                    "paraphrase_accepted": validation.paraphrase_accepted,
                    "overall_pass": validation.overall_pass,
                },
                "beta1": graph_stats.beta1,
                "cycles": graph_stats.cycles,
            }
            if chirality:
                record["chirality"] = {
                    "score": chirality.chirality_score,
                    "fisher_rao_distance": chirality.fisher_rao_distance,
                    "evidence_overlap": chirality.evidence_overlap,
                    "polarity_conflict": chirality.polarity_conflict,
                }
            results.append(record)
            sample_metrics.append(
                {
                    "claim_id": claim["id"],
                    "index": idx,
                    "timestamp": _iso_now(),
                    "schema_valid": validation.schema_valid,
                    "citation_valid": validation.citation_valid,
                    "entailment_score": validation.entailment_score,
                    "entailment_pass": validation.entailment_pass,
                    "semantic_similarity": validation.semantic_similarity,
                    "similarity_pass": validation.similarity_pass,
                    "paraphrase_accepted": validation.paraphrase_accepted,
                    "overall_pass": validation.overall_pass,
                    "beta1": graph_stats.beta1,
                    "chirality_score": chirality.chirality_score if chirality else None,
                    "fisher_rao_distance": chirality.fisher_rao_distance if chirality else None,
                    "evidence_overlap": chirality.evidence_overlap if chirality else evidence_overlap,
                    "chirality_polarity_conflict": bool(graph_stats.polarity_conflict),
                }
            )
            chirality_score = chirality.chirality_score if chirality else 0.0
            print(
                f"[eval]   finished sample {idx}/{total_samples} "
                f"| entailment={validation.entailment_score:.3f} "
                f"| beta1={graph_stats.beta1} "
                f"| chirality={chirality_score:.3f}",
                flush=True,
            )

        self._write_results(results)
        print(
            f"[eval] wrote {len(results)} completions to {self.config.output_path}",
            flush=True,
        )

        # Compute aggregate metrics
        total = metrics["total"] if metrics["total"] > 0 else 1  # Avoid division by zero

        # AGENTS.md Section 1.1 compliant metrics
        metrics["schema_compliance_rate"] = metrics["schema_valid"] / total
        metrics["citation_accuracy_rate"] = metrics["citation_valid"] / total
        metrics["mean_entailment_score"] = (
            sum(metrics["entailment_scores"]) / len(metrics["entailment_scores"])
            if metrics["entailment_scores"] else 0.0
        )
        metrics["entailment_pass_rate"] = metrics["entailment_pass"] / total
        metrics["mean_semantic_similarity"] = (
            sum(metrics["similarity_scores"]) / len(metrics["similarity_scores"])
            if metrics["similarity_scores"] else 0.0
        )
        metrics["semantic_similarity_rate"] = metrics["similarity_pass"] / total
        metrics["paraphrase_acceptance_rate"] = metrics["paraphrase_accepted"] / total
        metrics["overall_pass_rate"] = metrics["overall_pass"] / total

        # Legacy metrics (for comparison)
        metrics["c1_exact_match_rate_LEGACY"] = metrics["c1_exact_match"] / total
        metrics["evidence_exact_match_avg_LEGACY"] = (
            sum(metrics["evidence_exact_match"]) / len(metrics["evidence_exact_match"])
            if metrics["evidence_exact_match"] else 0.0
        )

        metrics["mean_beta1"] = (
            sum(metrics["beta1_values"]) / len(metrics["beta1_values"]) if metrics["beta1_values"] else 0.0
        )
        metrics["beta1_nonzero_rate"] = (
            metrics["beta1_nonzero"] / len(metrics["beta1_values"]) if metrics["beta1_values"] else 0.0
        )
        metrics["mean_chirality_score"] = (
            sum(metrics["chirality_scores"]) / len(metrics["chirality_scores"])
            if metrics["chirality_scores"] else 0.0
        )
        metrics["mean_fisher_rao_distance"] = (
            sum(metrics["fisher_rao_distances"]) / len(metrics["fisher_rao_distances"])
            if metrics["fisher_rao_distances"] else 0.0
        )
        metrics["std_entailment_score"] = pstdev(metrics["entailment_scores"]) if len(metrics["entailment_scores"]) >= 2 else 0.0
        metrics["std_semantic_similarity"] = (
            pstdev(metrics["similarity_scores"]) if len(metrics["similarity_scores"]) >= 2 else 0.0
        )

        # Print metrics aligned with AGENTS.md Section 1.1
        print("\n" + "=" * 80, flush=True)
        print("[eval] 4-STAGE SEMANTIC VALIDATION METRICS (AGENTS.md Section 1.1)", flush=True)
        print("=" * 80, flush=True)
        print(f"Total examples: {metrics['total']}", flush=True)
        print(f"\nSchema Compliance:     {metrics['schema_compliance_rate']:.1%} (target: ≥95%)", flush=True)
        print(f"Citation Accuracy:     {metrics['citation_accuracy_rate']:.1%} (hard gate)", flush=True)
        print(f"Mean Entailment Score: {metrics['mean_entailment_score']:.3f} (threshold: ≥0.75)", flush=True)
        print(f"Entailment Pass Rate:  {metrics['entailment_pass_rate']:.1%}", flush=True)
        print(f"Mean Similarity Score: {metrics['mean_semantic_similarity']:.3f} (threshold: ≥0.70)", flush=True)
        print(f"Similarity Pass Rate:  {metrics['semantic_similarity_rate']:.1%} (target: ≥60%)", flush=True)
        print(f"Paraphrase Accepted:   {metrics['paraphrase_acceptance_rate']:.1%}", flush=True)
        print(f"\n🎯 OVERALL PASS RATE:   {metrics['overall_pass_rate']:.1%}", flush=True)
        print("\nTopology / Chirality diagnostics:", flush=True)
        print(f"Mean β₁:               {metrics['mean_beta1']:.2f}", flush=True)
        print(f"β₁ > 0 rate:          {metrics['beta1_nonzero_rate']:.1%}", flush=True)
        print(f"Mean chirality score:  {metrics['mean_chirality_score']:.3f}", flush=True)
        print(f"Mean Fisher-Rao dist.: {metrics['mean_fisher_rao_distance']:.3f}", flush=True)
        print("\n" + "-" * 80, flush=True)
        print("LEGACY EXACT-MATCH METRICS (for comparison only, DO NOT optimize):", flush=True)
        print(f"C1 Exact Match:        {metrics['c1_exact_match_rate_LEGACY']:.1%}", flush=True)
        print(f"Evidence Exact Match:  {metrics['evidence_exact_match_avg_LEGACY']:.1%}", flush=True)
        print("=" * 80 + "\n", flush=True)

        metrics["per_sample_metrics"] = sample_metrics
        metrics["series"] = build_evaluation_series(sample_metrics)

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
        # Match training prompt format exactly (from scifact_claim_extractor.jsonl)
        prompt = "You are extracting atomic claims and their logical relations from scientific abstracts.\n\n"
        prompt += "Passage:\n"

        # Add document title and abstract (same format as training)
        for doc_id in claim.get("cited_doc_ids", []):
            doc = corpus.get(str(doc_id))
            if not doc:
                continue
            # Include Document ID so model knows which ID to cite
            prompt += f"Document {doc_id}: {doc['title']}\n\n"
            prompt += " ".join(doc["abstract"]) + "\n\n"

        # Add task instructions (same format as training)
        prompt += "Task:\n"
        prompt += "1. Restate the passage's central hypothesis verbatim (or with minimal edits) as CLAIM[c1].\n"
        prompt += "2. Continue listing distinct factual claims as CLAIM[c#] (Document <doc_id>): <text> using precise language from the passage.\n"
        prompt += "3. Use RELATION: <source_id> <supports|refutes> <target_id> to link evidence claims to the main hypothesis.\n\n"

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
