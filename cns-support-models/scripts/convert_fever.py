#!/usr/bin/env python3
"""
Convert FEVER claims + Wikipedia evidence into the claim-extractor JSONL format.

Usage:
    python scripts/convert_fever.py \
        --claims data/raw/fever/fever.train.jsonl \
        --wiki-dir data/raw/fever/wiki-pages \
        --out data/processed/fever_claim_extractor.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def load_wiki_sentences(wiki_dir: Path) -> Dict[Tuple[str, int], str]:
    """
    Build a lookup from (page_id, sentence_index) to sentence text.

    Supports both legacy TSV format and the JSONL format provided via Zenodo.
    """
    mapping: Dict[Tuple[str, int], str] = {}
    files = sorted([p for p in wiki_dir.rglob("*") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No wiki shard files under {wiki_dir}")

    for file in files:
        with file.open("r", encoding="utf-8") as fh:
            first = fh.readline()
            fh.seek(0)
            if not first:
                continue
            if first.lstrip().startswith("{"):
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    page = payload.get("id")
                    lines_blob = payload.get("lines", "")
                    if not page or not lines_blob:
                        continue
                    for row in lines_blob.split("\n"):
                        if not row:
                            continue
                        parts = row.split("\t")
                        if len(parts) < 2:
                            continue
                        idx_str, text = parts[0], parts[1]
                        if not text:
                            continue
                        try:
                            sent_idx = int(idx_str)
                        except ValueError:
                            continue
                        mapping[(page, sent_idx)] = text.strip()
            else:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 3:
                        continue
                    page, idx, text = parts[0], parts[1], parts[2]
                    try:
                        sent_idx = int(idx)
                    except ValueError:
                        continue
                    mapping[(page, sent_idx)] = text.strip()

    logging.info("Loaded %d wiki sentences from %d files", len(mapping), len(files))
    return mapping


def iter_claims(claims_path: Path) -> Iterable[dict]:
    with claims_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Skipping malformed claim line: %.50s", line)
                continue


def build_completion(claim: str, evidence_texts: List[str], label: str) -> str:
    lines = [f"CLAIM[c1]: {claim.strip()}"]
    for idx, text in enumerate(evidence_texts, start=2):
        cid = f"c{idx}"
        relation = "supports" if label.upper() == "SUPPORTS" else "refutes"
        lines.append(f"CLAIM[{cid}]: {text}")
        lines.append(f"RELATION: {cid} {relation} c1")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert FEVER to CNS training JSONL.")
    parser.add_argument("--claims", type=Path, required=True, help="fever.train.jsonl")
    parser.add_argument("--wiki-dir", type=Path, required=True, help="Directory with wiki shards")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--include-nei", action="store_true", help="Keep NOT ENOUGH INFO claims")
    args = parser.parse_args()

    wiki_lookup = load_wiki_sentences(args.wiki_dir)
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = defaultdict(int)
    with out_path.open("w", encoding="utf-8") as out_fh:
        for entry in iter_claims(args.claims):
            label = entry.get("label", "NOT ENOUGH INFO")
            if label == "NOT ENOUGH INFO" and not args.include_nei:
                skipped["nei"] += 1
                continue

            evidence_sets = entry.get("evidence") or []
            sentences: List[str] = []
            for ev_set in evidence_sets:
                for item in ev_set:
                    if len(item) < 4:
                        continue
                    _, _, page, sent_idx = item[:4]
                    text = wiki_lookup.get((page, sent_idx))
                    if text:
                        sentences.append(text)
                if sentences:
                    break  # take first valid evidence set

            if not sentences:
                skipped["missing_evidence"] += 1
                continue

            completion = build_completion(entry["claim"], sentences, label)
            prompt = (
                f"Passage (from {sentences[0][:32]}...):\n"
                f"{' '.join(sentences[:3])}\n\n"
                "Task: Extract atomic claims and relations.\n\n"
                "Output format:\nCLAIM[c#]: <text>\nRELATION: <src_id> <supports|refutes> <dst_id>\n\n"
            )
            payload = {
                "prompt": prompt,
                "completion": completion,
                "metadata": {
                    "source": "FEVER",
                    "label": label,
                    "id": entry.get("id"),
                },
            }
            out_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            kept += 1

    logging.info("Wrote %d FEVER examples to %s (skipped: %s)", kept, out_path, dict(skipped))


if __name__ == "__main__":
    main()
