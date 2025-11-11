from __future__ import annotations

import pytest

from scripts.claim_schema import (
    CLAIM_LINE_RE,
    RELATION_LINE_RE,
    enforce_c1,
    parse_claim_lines,
    parse_relation_line,
    render_claim_lines,
)


class TestClaimParsing:
    def test_parse_single_claim(self):
        lines = ["CLAIM[c1]: Vitamin C reduces cold duration."]
        claims = parse_claim_lines(lines)
        assert len(claims) == 1
        assert claims[0]["id"].lower() == "c1"
        assert "Vitamin C" in claims[0]["text"]

    def test_parse_multiple_claims_preserves_order(self):
        lines = [
            "CLAIM[c1]: Hypothesis text.",
            "CLAIM[c2]: Evidence sentence.",
            "CLAIM[c3]: Another evidence sentence.",
        ]
        claims = parse_claim_lines(lines)
        assert [c["id"] for c in claims] == ["c1", "c2", "c3"]

    def test_render_roundtrip(self):
        payload = [
            {"id": "c1", "text": "Hypothesis"},
            {"id": "c2", "text": "Evidence"},
        ]
        rendered = render_claim_lines(payload)
        reparsed = parse_claim_lines(rendered)
        assert reparsed == payload

    def test_enforce_c1_inserts_when_missing(self):
        claims = [{"id": "c2", "text": "Evidence only"}]
        enforced = enforce_c1(claims, "Canonical hypothesis")
        assert enforced[0]["id"] == "c1"
        assert enforced[0]["text"] == "Canonical hypothesis"

    @pytest.mark.parametrize(
        "line,expected",
        [
            ("CLAIM[ZX-10]: Text", True),
            ("Claim[c1]: text", True),
            ("CLAIM[]: missing", False),
        ],
    )
    def test_claim_regex_matches_expected(self, line: str, expected: bool):
        assert bool(CLAIM_LINE_RE.match(line)) is expected


class TestRelationParsing:
    def test_parse_supports_relation(self):
        rel = parse_relation_line("RELATION: c2 supports c1")
        assert rel == ("c2", "supports", "c1")

    def test_parse_refutes_relation_case_insensitive(self):
        rel = parse_relation_line("relation: C3 REFUTES C1")
        assert rel == ("C3", "refutes", "C1")

    def test_relation_regex_rejects_invalid(self):
        assert RELATION_LINE_RE.match("RELATION c2 -> c1") is None
