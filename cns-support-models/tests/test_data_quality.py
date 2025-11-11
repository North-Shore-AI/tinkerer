from __future__ import annotations

import json
from pathlib import Path

import copy

from scripts.claim_schema import parse_claim_lines, parse_relation_line
from scripts import validate_dataset


def _load_claim_map(claims_file: Path) -> dict[int, str]:
    claim_map: dict[int, str] = {}
    with claims_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            payload = json.loads(line)
            claim_map[int(payload["id"])] = payload["claim"].strip()
    return claim_map


def _load_dataset(dataset_file: Path) -> list[dict]:
    rows: list[dict] = []
    with dataset_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_c1_matches_source_claim(scifact_dataset_file: Path, scifact_claims_file: Path):
    claim_map = _load_claim_map(scifact_claims_file)
    rows = _load_dataset(scifact_dataset_file)

    for row in rows:
        claim_id = int(row["metadata"]["claim_id"])
        completion_lines = [l for l in row["completion"].splitlines() if l.startswith("CLAIM[")]
        claims = parse_claim_lines(completion_lines)
        assert claims[0]["id"].lower() == "c1"
        assert claims[0]["text"] == claim_map[claim_id]


def test_relations_reference_known_claims(scifact_dataset_file: Path):
    rows = _load_dataset(scifact_dataset_file)
    for row in rows:
        lines = row["completion"].splitlines()
        claims = parse_claim_lines(l for l in lines if l.startswith("CLAIM["))
        known_ids = {entry["id"].lower() for entry in claims}
        for line in lines:
            if not line.startswith("RELATION"):
                continue
            relation = parse_relation_line(line)
            assert relation is not None, f"Invalid relation line: {line}"
            src, _, dst = relation
            assert src.lower() in known_ids
            assert dst.lower() in known_ids


def test_claim_identifiers_are_sequential(scifact_dataset_file: Path):
    rows = _load_dataset(scifact_dataset_file)
    for row in rows:
        claims = parse_claim_lines(
            l for l in row["completion"].splitlines() if l.startswith("CLAIM[")
        )
        expected_idx = 1
        for entry in claims:
            assert entry["id"].lower() == f"c{expected_idx}"
            expected_idx += 1


def test_embedding_mode_allows_semantic_match(
    scifact_dataset_file: Path,
    scifact_claims_file: Path,
    scifact_corpus_file: Path,
):
    rows = _load_dataset(scifact_dataset_file)
    original_row = rows[0]
    modified = copy.deepcopy(original_row)
    lines = modified["completion"].splitlines()
    lines[1] = "CLAIM[c2]: Vitamin C supplements shorten common cold symptoms."
    modified["completion"] = "\n".join(lines)

    corpus_map = validate_dataset.load_corpus_map(scifact_corpus_file)
    claim_map, evidence_map = validate_dataset.load_claim_metadata(scifact_claims_file, corpus_map)

    # Exact mode should flag mismatch.
    exact_errors = validate_dataset.validate_row(
        1,
        modified,
        claim_map,
        evidence_map,
        evidence_mode="exact",
        embedding_matcher=None,
    )
    assert any("evidence text does not match" in err for err in exact_errors)

    # Embedding matcher with dummy encoder that treats cold-related sentences as similar.
    def dummy_encoder(texts):
        vectors = []
        for text in texts:
            vectors.append(
                [
                    1.0 if "vitamin" in text.lower() else 0.0,
                    1.0 if "cold" in text.lower() else 0.0,
                ]
            )
        return vectors

    matcher = validate_dataset.EmbeddingMatcher(
        "dummy",
        threshold=0.5,
        encoder=dummy_encoder,
    )
    embedding_errors = validate_dataset.validate_row(
        1,
        modified,
        claim_map,
        evidence_map,
        evidence_mode="embedding",
        embedding_matcher=matcher,
    )
    assert not embedding_errors
