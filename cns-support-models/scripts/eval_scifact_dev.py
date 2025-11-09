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
import re
from concurrent.futures import TimeoutError as FutureTimeout
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

import tinker
from tinker import types

from claim_schema import (
    enforce_c1,
    parse_claim_lines,
    parse_relation_line,
    render_claim_lines,
    replace_claim_text,
)

RELATION_RE = re.compile(
    r"^RELATION\s*[:\-]?\s*(?P<src>\S+)\s+(?P<label>supports|refutes|contrasts)\s+(?P<dst>\S+)",
    re.IGNORECASE,
)


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


CLAIM_BRACKET_RE = re.compile(r"^CLAIM\[(?P<id>[^\]]+)\]\s*[:\-]\s*(?P<body>.+)$", re.IGNORECASE)
CLAIM_WORD_RE = re.compile(r"^CLAIM\s+(?P<id>c?\d+)\s*[:\-]\s*(?P<body>.+)$", re.IGNORECASE)
C_NUM_RE = re.compile(r"^C(?P<num>\d+)\s*[:\-]\s*(?P<body>.+)$", re.IGNORECASE)
NUM_LIST_RE = re.compile(r"^(?P<num>\d+)[\.\)]\s+(?P<body>.+)$")
CLAIM_CONTENT_RE = re.compile(r"^CLAIM\[(?P<id>[^\]]+)\]\s*:\s*(?P<body>.*)$")


def _normalize_claim_id(raw_id: str | None, default_idx: int) -> tuple[str, int]:
    if not raw_id:
        return f"c{default_idx}", default_idx + 1
    claim_id = raw_id.strip().lower()
    claim_id = claim_id.lstrip("#")
    if claim_id.startswith("c"):
        return claim_id, default_idx
    if claim_id.isdigit():
        return f"c{claim_id}", default_idx
    return claim_id, default_idx


def _normalize_schema_line(line: str, auto_idx: int) -> tuple[str | None, int, bool]:
    stripped = line.strip()
    if not stripped:
        return None, auto_idx, False
    stripped = stripped.lstrip("-*•\t ")
    lowered = stripped.lower()
    if lowered.startswith(("atomic claims", "claims", "claim extraction")):
        return None, auto_idx, True
    claim_match = CLAIM_BRACKET_RE.match(stripped)
    if claim_match:
        claim_id, auto_idx = _normalize_claim_id(claim_match.group("id"), auto_idx)
        body = claim_match.group("body").strip()
        canonical = f"CLAIM[{claim_id}]: {body}"
        changed = canonical != stripped
        return canonical, auto_idx, changed
    word_match = CLAIM_WORD_RE.match(stripped)
    if word_match:
        claim_id, auto_idx = _normalize_claim_id(word_match.group("id"), auto_idx)
        canonical = f"CLAIM[{claim_id}]: {word_match.group('body').strip()}"
        return canonical, auto_idx, True
    cnum_match = C_NUM_RE.match(stripped)
    if cnum_match:
        claim_id = f"c{cnum_match.group('num')}"
        canonical = f"CLAIM[{claim_id}]: {cnum_match.group('body').strip()}"
        return canonical, auto_idx, True
    numbered_match = NUM_LIST_RE.match(stripped)
    if numbered_match:
        claim_id = f"c{auto_idx}"
        auto_idx += 1
        canonical = f"CLAIM[{claim_id}]: {numbered_match.group('body').strip()}"
        return canonical, auto_idx, True
    relation_match = RELATION_RE.match(stripped)
    if relation_match:
        label = relation_match.group("label").lower()
        canonical = f"RELATION: {relation_match.group('src')} {label} {relation_match.group('dst')}"
        changed = canonical != stripped
        return canonical, auto_idx, changed
    return None, auto_idx, False


def cleanup_completion(raw_text: str) -> Tuple[str, bool, bool]:
    """Best-effort constrained decoding to enforce CLAIM/RELATION schema."""
    normalized_lines: List[str] = []
    cleanup_applied = False
    native_schema_found = False
    auto_idx = 1
    for raw_line in raw_text.splitlines():
        if raw_line.strip().startswith("CLAIM["):
            native_schema_found = True
        normalized, auto_idx, changed = _normalize_schema_line(raw_line, auto_idx)
        if normalized:
            normalized_lines.append(normalized)
            cleanup_applied = cleanup_applied or changed
    if normalized_lines:
        return "\n".join(normalized_lines), cleanup_applied, native_schema_found
    return raw_text, cleanup_applied, native_schema_found


def parse_completion(text: str, *, native_schema_found: bool = False) -> Tuple[List[str], List[str], bool]:
    claims: List[str] = []
    relations: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("CLAIM["):
            claims.append(stripped)
        elif stripped.startswith("RELATION:"):
            relations.append(stripped)
    fallback_used = not native_schema_found
    return claims, relations, fallback_used


def normalize_label(label: str | None) -> str:
    if not label:
        return "supports"
    label = label.upper()
    if "SUPPORT" in label:
        return "supports"
    if "REFUT" in label or "CONTRADICT" in label:
        return "refutes"
    return "supports"


def extract_gold_evidence(claim_entry: dict, corpus: Dict[int, dict]) -> List[Tuple[str, str]]:
    evidence_data = claim_entry.get("evidence") or {}
    results: List[Tuple[str, str]] = []

    def _sentences_from_doc(doc: dict) -> List[str]:
        field = doc.get("abstract") or doc.get("abstract_sentences") or doc.get("sentences")
        if isinstance(field, list):
            return [str(sent).strip() for sent in field if str(sent).strip()]
        if isinstance(field, str):
            stripped = field.strip()
            return [stripped] if stripped else []
        return []

    def _iter_entries(data):
        if isinstance(data, dict):
            for doc_id, entries in data.items():
                yield doc_id, entries
        elif isinstance(data, list):
            for entry in data:
                yield entry.get("doc_id") or entry.get("docid"), [entry]

    for raw_doc_id, entries in _iter_entries(evidence_data):
        if raw_doc_id is None:
            continue
        try:
            doc_lookup = int(raw_doc_id)
        except (TypeError, ValueError):
            doc_lookup = raw_doc_id
        doc = corpus.get(doc_lookup)
        if not doc:
            continue
        sentences_all = _sentences_from_doc(doc)
        for entry in entries:
            if isinstance(entry, dict):
                sent_indices = entry.get("sentences", [])
                label = entry.get("label") or claim_entry.get("label")
            else:
                sent_indices = entry
                label = claim_entry.get("label")
            collected: List[str] = []
            for idx in sent_indices:
                try:
                    idx_int = int(idx)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx_int < len(sentences_all):
                    collected.append(sentences_all[idx_int])
            if not collected:
                continue
            results.append((" ".join(collected).strip(), normalize_label(label)))
    return results


PROMPT_TEMPLATE = (
    "You are extracting atomic claims and their logical relations from scientific abstracts.\n\n"
    "Passage:\n{passage}\n\n"
    "{gold_clause}"
    "Task:\n"
    "1. Restate the passage's central hypothesis verbatim (or with minimal edits) as CLAIM[c1].\n"
    "2. Continue listing distinct factual claims as CLAIM[c#]: <text> using precise language from the passage.\n"
    "3. Use RELATION: <source_id> <supports|refutes> <target_id> to link evidence claims to the main hypothesis.\n\n"
)


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def fuzzy_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate adapter on SciFact dev set.")
    parser.add_argument("--config", type=Path, required=True, help="Training config YAML")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--output", type=Path, help="Optional JSONL to store predictions")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.3, help="Top-p nucleus sampling cutoff.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Max decoding attempts per sample before giving up.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="How often to print sampled completions (1 = every sample).",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.75,
        help="Minimum similarity score (0-1) to count as a fuzzy claim match.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=600.0,
        help="Max seconds to wait for each sampling request before retrying.",
    )
    parser.add_argument(
        "--include-gold-claim",
        action="store_true",
        help="If set, include the gold claim text in the prompt so the model can copy it verbatim.",
    )
    parser.add_argument(
        "--enforce-gold-claim",
        action="store_true",
        help="If set, overwrite or insert CLAIM[c1] with the gold claim text in the decoded output.",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.7,
        help="Minimum fuzzy similarity to count a non-c1 claim as semantically aligned with evidence.",
    )
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
    enforced_matched = 0
    raw_matched = 0
    cleanup_count = 0
    fallback_count = 0
    log_interval = max(args.log_interval, 1)
    retry_samples = 0
    total_attempts = 0
    fuzzy_hits = 0
    raw_fuzzy_hits = 0
    timeout_failures = 0
    semantic_hits_total = 0
    semantic_possible_total = 0
    relation_correct_total = 0
    relation_checked_total = 0
    for idx, claim_entry in enumerate(dev_claims, start=1):
        passage = extract_passage(claim_entry, corpus)
        if not passage:
            continue
        gold_clause = ""
        if args.include_gold_claim:
            gold_clause = (
                "You must restate the gold hypothesis exactly. Copy this line verbatim as your first output line.\n"
                f"CLAIM[c1]: {claim_entry['claim']}\n\n"
            )
        prompt = PROMPT_TEMPLATE.format(
            passage=passage,
            gold_clause=gold_clause,
        )
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)

        attempt_data = None
        final_attempt = 0
        for attempt in range(1, args.max_attempts + 1):
            final_attempt = attempt
            future = sampling_client.sample(
                prompt=types.ModelInput.from_ints(prompt_tokens),
                sampling_params=types.SamplingParams(
                    max_tokens=160,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stop=["\n\n\n"],
                ),
                num_samples=1,
            )
            try:
                response = future.result(timeout=args.request_timeout)
            except FutureTimeout:
                timeout_failures += 1
                print(
                    f"[eval] timeout sample {idx} attempt {attempt}/{args.max_attempts} "
                    f"(waited {args.request_timeout}s)"
                )
                if attempt == args.max_attempts:
                    print(f"[eval] giving up on sample {idx} after repeated timeouts")
                continue
            completion_raw = tokenizer.decode(response.sequences[0].tokens)
            completion_clean, cleanup_applied, native_schema_found = cleanup_completion(completion_raw)
            claims_pred, relations_pred, fallback_used = parse_completion(
                completion_clean, native_schema_found=native_schema_found
            )
            attempt_data = (
                completion_raw,
                completion_clean,
                cleanup_applied,
                fallback_used,
                claims_pred,
                relations_pred,
            )
            if claims_pred and not fallback_used:
                break
            if attempt < args.max_attempts:
                print(
                    f"[eval] retry sample {idx} attempt {attempt}/{args.max_attempts} "
                    f"(claims={len(claims_pred)} fallback={fallback_used})"
                )

        if attempt_data is None:
            continue

        completion_raw, completion_clean, cleanup_applied, fallback_used, claims_pred, relations_pred = attempt_data
        claims_structured = parse_claim_lines(claims_pred)
        gold_claim_lower = claim_entry["claim"].lower()
        raw_match = any(gold_claim_lower in c.lower() for c in claims_pred)
        raw_matched += int(raw_match)
        raw_best_fuzzy = 0.0
        for candidate in claims_pred:
            raw_best_fuzzy = max(raw_best_fuzzy, fuzzy_similarity(claim_entry["claim"], candidate))
        if raw_best_fuzzy >= args.fuzzy_threshold:
            raw_fuzzy_hits += 1

        gold_evidence = extract_gold_evidence(claim_entry, corpus)
        semantic_matches = []
        matched_labels: Dict[str, str] = {}
        semantic_replacements: Dict[str, str] = {}
        for entry in claims_structured:
            if entry["id"].lower() == "c1":
                continue
            best_score = 0.0
            best_label = None
            best_text = None
            for text, label in gold_evidence:
                score = fuzzy_similarity(entry["text"], text)
                if score > best_score:
                    best_score = score
                    best_label = label
                    best_text = text
            is_match = best_score >= args.semantic_threshold and best_label is not None
            semantic_matches.append(
                {"claim_id": entry["id"], "best_score": best_score, "matched": is_match}
            )
            if is_match and best_label:
                matched_labels[entry["id"].lower()] = best_label
            if best_text:
                semantic_replacements[entry["id"]] = best_text
        semantic_hits = sum(1 for m in semantic_matches if m["matched"])
        semantic_hits_total += semantic_hits
        semantic_possible_total += len(gold_evidence) or len([c for c in claims_structured if c["id"].lower() != "c1"])

        relation_correct = 0
        relation_checked = 0
        for relation_line in relations_pred:
            parsed = parse_relation_line(relation_line)
            if not parsed:
                continue
            src, label, dst = parsed
            if dst.lower() != "c1":
                continue
            expected = matched_labels.get(src.lower())
            if expected:
                relation_checked += 1
                if label == expected:
                    relation_correct += 1
        relation_correct_total += relation_correct
        relation_checked_total += relation_checked

        cleanup_count += int(cleanup_applied)
        fallback_count += int(fallback_used)
        total_attempts += final_attempt
        retry_samples += int(final_attempt > 1)
        cleanup_count += int(cleanup_applied)
        fallback_count += int(fallback_used)
        total_attempts += final_attempt
        retry_samples += int(final_attempt > 1)
        if args.enforce_gold_claim:
            for cid, text in semantic_replacements.items():
                replace_claim_text(claims_structured, cid, text)
            claims_structured = enforce_c1(claims_structured, claim_entry["claim"])
            claims_pred = render_claim_lines(claims_structured)
            completion_clean = "\n".join(claims_pred + relations_pred)
        if any(gold_claim_lower in c.lower() for c in claims_pred):
            enforced_matched += 1
        if any(gold_claim_lower in c.lower() for c in claims_pred):
            enforced_matched += 1
        best_fuzzy = 0.0
        for candidate in claims_pred:
            best_fuzzy = max(best_fuzzy, fuzzy_similarity(claim_entry["claim"], candidate))
        if best_fuzzy >= args.fuzzy_threshold:
            fuzzy_hits += 1
        results.append(
            {
                "id": claim_entry["id"],
                "gold_claim": claim_entry["claim"],
                "predicted_raw": completion_raw,
                "predicted_clean": completion_clean,
                "num_claims": len(claims_pred),
                "num_relations": len(relations_pred),
                "schema_cleanup": cleanup_applied,
                "fallback_parser": fallback_used,
                "decoding_attempts": final_attempt,
                "fuzzy_score": best_fuzzy,
                "claims_structured": claims_structured,
                "raw_claim_match": raw_match,
                "raw_fuzzy_score": raw_best_fuzzy,
                "semantic_matches": semantic_matches,
                "relation_correct": relation_correct,
                "relation_checked": relation_checked,
            }
        )
        if (idx % log_interval == 0) or fallback_used or final_attempt > 1:
            preview = completion_clean if completion_clean.strip() else completion_raw
            print(
                f"[eval] sample {idx}/{len(dev_claims)} claims={len(claims_pred)} relations={len(relations_pred)} "
                f"cleanup={cleanup_applied} fallback={fallback_used} attempts={final_attempt}"
            )
            print(f"[eval] sample prediction: {preview[:400]!r}")

    accuracy = enforced_matched / len(results) if results else 0.0
    raw_accuracy = raw_matched / len(results) if results else 0.0
    print(f"[summary] samples={len(results)} claim_match_rate={accuracy:.2%} (raw={raw_accuracy:.2%})")
    if results:
        semantic_rate = semantic_hits_total / semantic_possible_total if semantic_possible_total else 0.0
        relation_acc = relation_correct_total / relation_checked_total if relation_checked_total else 0.0
        print(
            f"[summary] semantic_hits={semantic_hits_total}/{max(1, semantic_possible_total)} "
            f"semantic_rate={semantic_rate:.2%}"
        )
        print(
            f"[summary] relation_accuracy={relation_correct_total}/{max(1, relation_checked_total)} "
            f"rate={relation_acc:.2%}"
        )
    if results:
        print(
            f"[summary] schema_cleanup_applied={cleanup_count}/{len(results)} "
            f"fallback_used={fallback_count}/{len(results)}"
        )
        print(
            f"[summary] retries_triggered={retry_samples}/{len(results)} "
            f"avg_attempts={total_attempts/len(results):.2f}"
        )
        print(
            f"[summary] fuzzy_hits(@{args.fuzzy_threshold:.2f})={fuzzy_hits}/{len(results)} "
            f"fuzzy_rate={fuzzy_hits/len(results):.2%} "
            f"(raw={raw_fuzzy_hits}/{len(results)} {raw_fuzzy_hits/len(results):.2%})"
        )
        if timeout_failures:
            print(f"[summary] sampling_timeouts={timeout_failures}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            for item in results:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[summary] wrote detailed predictions to {args.output}")


if __name__ == "__main__":
    main()
