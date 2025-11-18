#!/usr/bin/env python3
"""
Quick helper to visualize CLAIM/RELATION graphs and compute β₁.

Usage:
    python scripts/build_graph.py --dataset cns-support-models/data/processed/scifact_claim_extractor.jsonl --line 1
    python scripts/build_graph.py --text completion.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from thinker.claim_schema import parse_claim_lines, parse_relation_line
from thinker.logic import compute_graph_stats


def _load_completion_from_dataset(path: Path, line_no: int) -> str:
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            if idx == line_no:
                payload = json.loads(line)
                completion = payload.get("completion")
                if not isinstance(completion, str):
                    raise ValueError(f"Row {line_no} missing 'completion' field")
                return completion
    raise ValueError(f"Line {line_no} not found in {path}")


def _load_completion_from_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_claim_ids(completion: str) -> List[str]:
    claims = parse_claim_lines(line for line in completion.splitlines() if line.strip().upper().startswith("CLAIM["))
    return list(claims.keys())


def _extract_relations(completion: str):
    relations = []
    for line in completion.splitlines():
        relation = parse_relation_line(line)
        if relation:
            relations.append(relation)
    return relations


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a reasoning graph from CLAIM/RELATION completions.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", type=Path, help="JSONL file containing prompt/completion rows.")
    group.add_argument("--text", type=Path, help="Plaintext file containing a completion.")
    parser.add_argument("--line", type=int, default=1, help="1-based row index when using --dataset (default: 1).")
    args = parser.parse_args()

    if args.dataset:
        completion = _load_completion_from_dataset(args.dataset, args.line)
    else:
        completion = _load_completion_from_text(args.text)

    claim_ids = _extract_claim_ids(completion)
    relations = _extract_relations(completion)
    stats = compute_graph_stats(claim_ids, relations)

    print(f"Nodes:       {stats.nodes}")
    print(f"Edges:       {stats.edges}")
    print(f"Components:  {stats.components}")
    print(f"β₁ (cycles): {stats.beta1}")
    print(f"Polarity conflict on c1: {'YES' if stats.polarity_conflict else 'no'}")
    if stats.cycles:
        print("\nCycles detected:")
        for cycle in stats.cycles:
            print("  -> ".join(cycle))
    else:
        print("\nNo cycles detected.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
