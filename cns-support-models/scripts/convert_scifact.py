#!/usr/bin/env python3
"""
Convert the public SciFact dataset into the JSONL format expected by the claim extractor.

Usage:
    python convert_scifact.py \
        --claims_json /path/to/claims_dev.jsonl \
        --corpus_json /path/to/corpus.jsonl \
        --out data/processed/scifact_claim_extractor.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

Relation = Tuple[str, str, str]  # (source_id, relation_type, target_id)


def load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def normalize_label(label: str | None) -> str:
    if not label:
        return "supports"
    label = label.upper()
    if "SUPPORT" in label:
        return "supports"
    if "REFUT" in label or "CONTRADICT" in label:
        return "refutes"
    return "supports"


def build_passage(documents: List[dict]) -> str:
    blocks: List[str] = []
    for doc in documents:
        title = doc.get("title")
        if title:
            blocks.append(title.strip())
        abstract = doc.get("abstract") or doc.get("abstract_sentences") or doc.get("sentences")
        if isinstance(abstract, list):
            blocks.append(" ".join(sent.strip() for sent in abstract))
        elif isinstance(abstract, str):
            blocks.append(abstract.strip())
    return "\n\n".join(blocks).strip()


def gather_evidence(claim_entry: dict) -> Iterable[dict]:
    """
    Normalize SciFact evidence format into dicts with docid, sentences, and label.

    Evidence can appear as:
        {doc_id: [[sent_idx], ...]}
        {doc_id: [{'sentences': [...], 'label': 'SUPPORTS'}]}
    """
    evidence = claim_entry.get("evidence", {})
    if isinstance(evidence, dict):
        for doc_id, entries in evidence.items():
            for entry in entries:
                if isinstance(entry, dict):
                    sentences = entry.get("sentences", [])
                    label = entry.get("label") or claim_entry.get("label")
                else:
                    sentences = entry
                    label = claim_entry.get("label")
                yield dict(docid=str(doc_id), sentences=sentences, label=label)
    elif isinstance(evidence, list):
        for ev in evidence:
            yield ev


def build_claim_completion(claim_text: str, evidence_texts: List[Tuple[str, Sequence[str], str]]) -> str:
    lines = [f"CLAIM[c1]: {claim_text.strip()}"]
    claim_idx = 2
    for _, sentences, label in evidence_texts:
        if not sentences:
            continue
        cid = f"c{claim_idx}"
        lines.append(f"CLAIM[{cid}]: {' '.join(sentences).strip()}")
        relation = normalize_label(label)
        lines.append(f"RELATION: {cid} {relation} c1")
        claim_idx += 1
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SciFact into claim-extractor JSONL format.")
    parser.add_argument("--claims_json", type=Path, required=True, help="claims_dev.jsonl or claims_train.jsonl")
    parser.add_argument("--corpus_json", type=Path, required=True, help="corpus.jsonl with abstracts")
    parser.add_argument("--out", type=Path, required=True, help="Destination JSONL file")
    args = parser.parse_args()

    corpus_entries = load_jsonl(args.corpus_json)
    corpus_map: Dict[str, dict] = {}
    for entry in corpus_entries:
        doc_id = str(entry.get("doc_id") or entry.get("docid") or entry.get("id"))
        if not doc_id:
            continue
        corpus_map[doc_id] = entry

    claims = load_jsonl(args.claims_json)
    output_path = args.out
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_no_docs = 0
    skipped_no_evidence = 0
    with output_path.open("w", encoding="utf-8") as out_fh:
        for claim in claims:
            claim_text = claim.get("claim")
            if not claim_text:
                skipped_no_docs += 1
                continue
            evidence_entries = list(gather_evidence(claim)) or []
            doc_ids = {str(ev.get("doc_id") or ev.get("docid")) for ev in evidence_entries if ev.get("docid") or ev.get("doc_id")}
            docs = [corpus_map[doc_id] for doc_id in doc_ids if doc_id in corpus_map]
            if not docs:
                skipped_no_docs += 1
                continue
            prompt = (
                "You are extracting atomic claims and their logical relations from scientific abstracts.\n\n"
                "Passage:\n"
                f"{build_passage(docs)}\n\n"
                "Task:\n"
                "1. List distinct factual claims as CLAIM[c#]: <text>.\n"
                "2. Use RELATION: <source_id> <supports|refutes> <target_id> to link evidence claims to the main hypothesis.\n\n"
            )

            evidence_texts: List[Tuple[str, Sequence[str], str]] = []
            for entry in evidence_entries:
                doc_id = str(entry.get("doc_id") or entry.get("docid"))
                doc = corpus_map.get(doc_id)
                if not doc:
                    continue
                sentences_field = doc.get("abstract") or doc.get("abstract_sentences") or doc.get("sentences")
                if isinstance(sentences_field, list):
                    indices = []
                    for idx in entry.get("sentences", []):
                        try:
                            indices.append(int(idx))
                        except (TypeError, ValueError):
                            continue
                    sentences = [
                        sentences_field[idx].strip()
                        for idx in indices
                        if 0 <= idx < len(sentences_field)
                    ]
                else:
                    sentences = [doc.get("abstract", "")]
                evidence_texts.append((doc_id, sentences, entry.get("label") or claim.get("label")))

            evidence_texts = [(doc_id, sentences, label) for doc_id, sentences, label in evidence_texts if sentences]
            if not evidence_texts:
                skipped_no_evidence += 1
                continue

            completion = build_claim_completion(claim_text, evidence_texts)
            payload = {
                "prompt": prompt,
                "completion": completion.strip(),
                "metadata": {
                    "source": "SciFact",
                    "claim_id": claim.get("id"),
                    "doc_ids": list(doc_ids),
                    "license": "CC-BY-4.0",
                },
            }
            out_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1

    print(
        f"Wrote {written} examples to {output_path} "
        f"(skipped_no_docs={skipped_no_docs}, skipped_no_evidence={skipped_no_evidence})"
    )


if __name__ == "__main__":
    main()
