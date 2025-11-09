#!/usr/bin/env python3
"""
Convert a manually annotated CSV into the JSONL format expected by the claim extractor.

CSV schema:
    passage,text
    claims,"CLAIM[c1]: ... | CLAIM[c2]: ..."
    relations,"c2 supports c1; c3 refutes c1"

Usage:
    python csv_to_claim_jsonl.py --csv data/annotated.csv --out data/processed/manual_claims.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_list(field: str | None, sep: str) -> list[str]:
    if not field:
        return []
    return [item.strip() for item in field.split(sep) if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV -> JSONL converter for claim extraction data.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to annotated CSV file")
    parser.add_argument("--out", type=Path, required=True, help="Destination JSONL path")
    args = parser.parse_args()

    output_path = args.out
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with args.csv.open("r", encoding="utf-8") as csv_fh, output_path.open("w", encoding="utf-8") as out_fh:
        reader = csv.DictReader(csv_fh)
        for row in reader:
            passage = row.get("passage") or row.get("text") or ""
            claims_field = row.get("claims") or ""
            relations_field = row.get("relations") or ""

            claims = parse_list(claims_field, "|")
            relations = parse_list(relations_field, ";")

            prompt = (
                "Extract atomic claims and their relations from the following passage. "
                "Return them in the CLAIM[...] and RELATION: format.\n\n"
                f"Passage:\n{passage.strip()}\n\n"
            )

            completion_lines = []
            for claim in claims:
                completion_lines.append(claim.strip())
            for rel in relations:
                completion_lines.append(f"RELATION: {rel}")

            payload = {
                "prompt": prompt,
                "completion": "\n".join(completion_lines),
                "metadata": {
                    "source": row.get("source", "manual"),
                    "annotator": row.get("annotator", ""),
                },
            }
            out_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"Converted CSV -> {output_path}")


if __name__ == "__main__":
    main()
