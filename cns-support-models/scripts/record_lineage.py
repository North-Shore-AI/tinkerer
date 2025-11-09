#!/usr/bin/env python3
"""
Record SHA-256 lineage for raw and processed dataset artifacts.

Usage:
    python scripts/record_lineage.py \
        --files data/raw/scifact/claims_train.jsonl \
                 data/raw/scifact/corpus.jsonl \
                 data/processed/scifact_claim_extractor.jsonl \
        --out data/lineage/scifact.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute SHA-256 lineage records.")
    parser.add_argument("--files", nargs="+", required=True, help="Files to hash")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path")
    args = parser.parse_args()

    records: Dict[str, dict] = {}
    for file_path in args.files:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        records[str(path)] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)
    print(f"[lineage] wrote hashes for {len(records)} files to {args.out}")


if __name__ == "__main__":
    main()
