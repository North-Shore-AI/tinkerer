#!/usr/bin/env python3
"""
Validate claim-extractor training JSONL files before running expensive jobs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

import math

try:
    from .claim_schema import parse_claim_lines, parse_relation_line
except ImportError:  # pragma: no cover - fallback for direct execution
    from claim_schema import parse_claim_lines, parse_relation_line


def ensure_readable(path: Path | None, description: str) -> Path | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found: {path}. Did you run 'make scifact' or generate the dataset?"
        )
    return path


def load_corpus_map(path: Path | None) -> dict[str, Sequence[str]]:
    if path is None:
        return {}
    path = ensure_readable(path, "Corpus JSONL")
    corpus: dict[str, Sequence[str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            payload = json.loads(line)
            doc_id = str(payload.get("doc_id") or payload.get("docid") or payload.get("id"))
            if not doc_id:
                continue
            abstract = payload.get("abstract") or payload.get("abstract_sentences") or payload.get("sentences") or []
            if isinstance(abstract, list):
                corpus[doc_id] = [sent.strip() for sent in abstract if isinstance(sent, str)]
            elif isinstance(abstract, str):
                corpus[doc_id] = [abstract.strip()]
    return corpus


def load_claim_metadata(
    claims_path: Path | None,
    corpus_map: dict[str, Sequence[str]] | None = None,
) -> Tuple[dict[int, str], dict[int, List[str]]]:
    if claims_path is None:
        return {}, {}
    claims_path = ensure_readable(claims_path, "Claims JSONL")
    corpus_map = corpus_map or {}
    claim_map: dict[int, str] = {}
    evidence_map: dict[int, List[str]] = {}
    with claims_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            payload = json.loads(line)
            claim_id = int(payload["id"])
            claim_map[claim_id] = payload["claim"].strip()
            evidence_entries = payload.get("evidence", {})
            sentences: List[str] = []
            if isinstance(evidence_entries, dict):
                entries = evidence_entries.items()
            else:
                entries = []
            for raw_doc_id, doc_entries in entries:
                doc_id = str(raw_doc_id)
                doc_sentences = corpus_map.get(doc_id, [])
                for entry in doc_entries:
                    indices = entry.get("sentences", []) if isinstance(entry, dict) else entry
                    for idx_val in indices:
                        try:
                            idx_int = int(idx_val)
                        except (TypeError, ValueError):
                            continue
                        if 0 <= idx_int < len(doc_sentences):
                            sentences.append(doc_sentences[idx_int])
            if sentences:
                evidence_map[claim_id] = sentences
    return claim_map, evidence_map


def iter_dataset(path: Path, max_examples: int | None = None) -> Iterable[Tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            if max_examples and idx > max_examples:
                break
            yield idx, json.loads(line)


class EmbeddingMatcher:
    """Utility for embedding-based evidence matching."""

    def __init__(
        self,
        model_name: str,
        threshold: float,
        encoder: Callable[[List[str]], Sequence[Sequence[float]]] | None = None,
    ):
        self.threshold = threshold
        if encoder is not None:
            self._encode = encoder
            self._uses_external_model = False
        else:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers must be installed to use embedding mode"
                ) from exc
            self._model = SentenceTransformer(model_name)
            self._encode = self._model.encode
            self._uses_external_model = True

    def _normalize(self, matrix: Sequence[Sequence[float]]) -> List[List[float]]:
        normalized: List[List[float]] = []
        for row in matrix:
            norm = math.sqrt(sum(x * x for x in row)) or 1e-8
            normalized.append([x / norm for x in row])
        return normalized

    def batch_match(self, candidates: List[str], gold: List[str]) -> List[bool]:
        if not gold:
            return [False] * len(candidates)
        cand_emb = self._normalize(self._encode(candidates))
        gold_emb = self._normalize(self._encode(gold))
        results: List[bool] = []
        for cand_vec in cand_emb:
            best = -1.0
            for gold_vec in gold_emb:
                score = sum(c * g for c, g in zip(cand_vec, gold_vec))
                if score > best:
                    best = score
            results.append(best >= self.threshold)
        return results


def validate_row(
    idx: int,
    row: dict,
    claim_map: dict[int, str],
    evidence_map: dict[int, List[str]],
    *,
    evidence_mode: str,
    embedding_matcher: EmbeddingMatcher | None = None,
) -> List[str]:
    errors: List[str] = []
    prompt = row.get("prompt")
    completion = row.get("completion")
    metadata = row.get("metadata", {})
    if not isinstance(prompt, str) or not prompt.strip():
        errors.append("prompt must be a non-empty string")
    if not isinstance(completion, str) or not completion.strip():
        errors.append("completion must be a non-empty string")
        return errors

    claim_lines = [line for line in completion.splitlines() if line.startswith("CLAIM[")]
    if not claim_lines:
        errors.append("completion missing CLAIM lines")
        return errors

    claims = parse_claim_lines(claim_lines)
    if not claims or claims[0]["id"].lower() != "c1":
        errors.append("first claim must be CLAIM[c1]")

    claim_id = metadata.get("claim_id")
    if claim_map and claim_id in claim_map:
        gold = claim_map[claim_id]
        if claims[0]["text"] != gold:
            errors.append("CLAIM[c1] does not match gold claim text")

    known_ids = {entry["id"].lower() for entry in claims}
    for line in completion.splitlines():
        if not line.startswith("RELATION"):
            continue
        relation = parse_relation_line(line)
        if not relation:
            errors.append(f"invalid relation syntax: {line}")
            continue
        src, _, dst = relation
        if src.lower() not in known_ids or dst.lower() not in known_ids:
            errors.append(f"relation references unknown claims: {line}")

    for expected, entry in enumerate(claims, start=1):
        if entry["id"].lower() != f"c{expected}":
            errors.append(f"claim IDs must be sequential (found {entry['id']})")
            break

    gold_evidence = evidence_map.get(claim_id) if claim_id is not None else None
    evidence_claims = [entry["text"] for entry in claims[1:]]
    if evidence_claims and gold_evidence:
        if evidence_mode == "exact":
            for text in evidence_claims:
                if text not in gold_evidence:
                    errors.append("evidence text does not match any gold sentence (exact)")
        elif evidence_mode == "embedding":
            if embedding_matcher is None:
                errors.append("embedding matcher not configured")
            else:
                results = embedding_matcher.batch_match(evidence_claims, gold_evidence)
                for matched, text in zip(results, evidence_claims):
                    if not matched:
                        errors.append(
                            f"evidence text failed embedding match (threshold={embedding_matcher.threshold}): {text}"
                        )
        else:
            errors.append(f"unknown evidence mode '{evidence_mode}'")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate claim-extractor datasets.")
    parser.add_argument("dataset", type=Path, help="JSONL dataset file")
    parser.add_argument(
        "--claims-json",
        type=Path,
        help="Optional SciFact claims JSONL for verifying CLAIM[c1] text",
    )
    parser.add_argument(
        "--corpus-json",
        type=Path,
        help="SciFact corpus JSONL required for evidence validation when using embeddings",
    )
    parser.add_argument(
        "--evidence-mode",
        choices=("exact", "embedding"),
        default="exact",
        help="How to compare evidence sentences to gold references",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformers model to use when --evidence-mode=embedding",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.75,
        help="Cosine similarity threshold for embedding matches",
    )
    parser.add_argument("--max-examples", type=int, default=None, help="Limit rows to inspect")
    args = parser.parse_args()

    try:
        dataset_path = ensure_readable(args.dataset, "Dataset JSONL")
        corpus_map = load_corpus_map(args.corpus_json)
        claim_map, evidence_map = load_claim_metadata(args.claims_json, corpus_map)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    embedding_matcher: EmbeddingMatcher | None = None
    if args.evidence_mode == "embedding":
        if not claim_map or not evidence_map:
            print("Embedding mode requires --claims-json and --corpus-json", file=sys.stderr)
            return 1
        try:
            embedding_matcher = EmbeddingMatcher(
                args.embedding_model,
                args.similarity_threshold,
            )
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            print("Install sentence-transformers or choose --evidence-mode exact.", file=sys.stderr)
            return 1

    errors = []
    for idx, row in iter_dataset(dataset_path, args.max_examples):
        row_errors = validate_row(
            idx,
            row,
            claim_map,
            evidence_map,
            evidence_mode=args.evidence_mode,
            embedding_matcher=embedding_matcher,
        )
        if row_errors:
            for err in row_errors:
                errors.append(f"line {idx}: {err}")

    if errors:
        print("Dataset validation failed:")
        for err in errors[:50]:
            print(f"  - {err}")
        print(f"(showing {min(len(errors), 50)} of {len(errors)} errors)")
        return 1

    print(f"{args.dataset} passed validation ({args.max_examples or 'all'} examples checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
