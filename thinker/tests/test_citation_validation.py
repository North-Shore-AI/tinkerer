"""Tests for citation validation utilities."""

import pytest

from thinker.citation_validation import (
    CitationValidationResult,
    extract_document_ids,
    validate_citations,
    compute_citation_penalty,
    batch_validate_citations,
    citation_validation_stats,
)


class TestExtractDocumentIds:
    """Test document ID extraction."""

    def test_extract_single_document(self):
        """Test extracting a single document ID."""
        text = "Document 12345: This is some evidence."
        result = extract_document_ids(text)
        assert result == {"12345"}

    def test_extract_multiple_documents(self):
        """Test extracting multiple document IDs."""
        text = """
        Document 12345: First evidence.
        Document 67890: Second evidence.
        Document 11111: Third evidence.
        """
        result = extract_document_ids(text)
        assert result == {"12345", "67890", "11111"}

    def test_extract_claim_format(self):
        """Test extracting from CLAIM format."""
        text = "CLAIM[c1] (Document 12345): The claim text."
        result = extract_document_ids(text)
        assert result == {"12345"}

    def test_extract_parenthesis_format(self):
        """Test extracting from parenthesis format."""
        text = "Evidence from (Document 12345) shows that..."
        result = extract_document_ids(text)
        assert result == {"12345"}

    def test_extract_mixed_formats(self):
        """Test extracting from mixed formats."""
        text = """
        Document 12345: Introduction
        CLAIM[c1] (Document 67890): A claim
        Reference to (Document 11111) in text
        """
        result = extract_document_ids(text)
        assert result == {"12345", "67890", "11111"}

    def test_extract_eight_digit_ids(self):
        """Test extracting 8-digit document IDs (real SciFact format)."""
        text = "Document 38485364: Real evidence from SciFact."
        result = extract_document_ids(text)
        assert result == {"38485364"}

    def test_extract_no_documents(self):
        """Test extracting from text with no document IDs."""
        text = "This text has no document citations."
        result = extract_document_ids(text)
        assert result == set()

    def test_extract_case_insensitive(self):
        """Test that extraction is case-insensitive."""
        text = "document 12345: lowercase"
        result = extract_document_ids(text)
        assert result == {"12345"}

    def test_extract_duplicates(self):
        """Test that duplicates are deduplicated."""
        text = """
        Document 12345: First mention
        Document 12345: Second mention
        CLAIM[c1] (Document 12345): Third mention
        """
        result = extract_document_ids(text)
        assert result == {"12345"}


class TestValidateCitations:
    """Test citation validation."""

    def test_valid_single_citation(self):
        """Test validating a single valid citation."""
        prompt = "Document 12345: Evidence text"
        completion = "CLAIM[c1] (Document 12345): A claim based on evidence"

        result = validate_citations(prompt, completion)

        assert result.is_valid is True
        assert result.cited_docs == {"12345"}
        assert result.valid_docs == {"12345"}
        assert result.invalid_docs == set()
        assert result.hallucination_count == 0

    def test_valid_multiple_citations(self):
        """Test validating multiple valid citations."""
        prompt = """
        Document 12345: First evidence
        Document 67890: Second evidence
        """
        completion = """
        CLAIM[c1] (Document 12345): First claim
        CLAIM[c2] (Document 67890): Second claim
        """

        result = validate_citations(prompt, completion)

        assert result.is_valid is True
        assert result.cited_docs == {"12345", "67890"}
        assert result.valid_docs == {"12345", "67890"}
        assert result.invalid_docs == set()
        assert result.hallucination_count == 0

    def test_invalid_single_hallucination(self):
        """Test detecting a single hallucinated citation."""
        prompt = "Document 12345: Real evidence"
        completion = "CLAIM[c1] (Document 99999): Hallucinated claim"

        result = validate_citations(prompt, completion)

        assert result.is_valid is False
        assert result.cited_docs == {"99999"}
        assert result.valid_docs == {"12345"}
        assert result.invalid_docs == {"99999"}
        assert result.hallucination_count == 1

    def test_invalid_mixed_citations(self):
        """Test detecting mixed valid and invalid citations."""
        prompt = "Document 12345: Real evidence"
        completion = """
        CLAIM[c1] (Document 12345): Valid claim
        CLAIM[c2] (Document 99999): Hallucinated claim
        CLAIM[c3] (Document 88888): Another hallucination
        """

        result = validate_citations(prompt, completion)

        assert result.is_valid is False
        assert result.cited_docs == {"12345", "99999", "88888"}
        assert result.valid_docs == {"12345"}
        assert result.invalid_docs == {"99999", "88888"}
        assert result.hallucination_count == 2

    def test_real_world_claim_133(self):
        """Test with real data from HIGH severity claim 133."""
        prompt = """
        Document 38485364: The adaptor protein Tks5/Fish is required...
        Document 6969753: Dynamic interactions of cortactin...
        Document 17934082: Membrane lipids in invadopodia and podosomes...
        """
        # Claim 133 hallucinated documents 16280642 and 12640810
        completion = """
        CLAIM[c2] (Document 16280642): N-WASP bound all SH3 domains...
        CLAIM[c3] (Document 12640810): Cortactin phosphorylation...
        """

        result = validate_citations(prompt, completion)

        assert result.is_valid is False
        assert result.valid_docs == {"38485364", "6969753", "17934082"}
        assert "16280642" in result.invalid_docs
        assert "12640810" in result.invalid_docs
        assert result.hallucination_count == 2

    def test_real_world_claim_179(self):
        """Test with real data from HIGH severity claim 179."""
        prompt = """
        Document 16322674: Birth Size and Breast Cancer Risk...
        Document 27123743: Role of birthweight in the etiology...
        Document 23557241: Intrauterine factors and risk...
        """
        # Claim 179 hallucinated document 17450673
        completion = """
        CLAIM[c2] (Document 17450673): We found that heavier birth weights...
        """

        result = validate_citations(prompt, completion)

        assert result.is_valid is False
        assert result.valid_docs == {"16322674", "27123743", "23557241"}
        assert result.invalid_docs == {"17450673"}
        assert result.hallucination_count == 1

    def test_no_citations_in_completion(self):
        """Test when completion has no citations."""
        prompt = "Document 12345: Evidence"
        completion = "This is a completion with no citations."

        result = validate_citations(prompt, completion)

        assert result.is_valid is True  # No citations = valid (not hallucinating)
        assert result.cited_docs == set()
        assert result.hallucination_count == 0


class TestComputeCitationPenalty:
    """Test citation penalty computation."""

    def test_penalty_zero_for_valid(self):
        """Test that valid citations have zero penalty."""
        result = CitationValidationResult(
            is_valid=True,
            cited_docs={"12345"},
            valid_docs={"12345"},
            invalid_docs=set(),
            hallucination_count=0,
        )
        penalty = compute_citation_penalty(result, penalty_weight=2.0)
        assert penalty == 0.0

    def test_penalty_single_hallucination(self):
        """Test penalty for single hallucination."""
        result = CitationValidationResult(
            is_valid=False,
            cited_docs={"99999"},
            valid_docs={"12345"},
            invalid_docs={"99999"},
            hallucination_count=1,
        )
        penalty = compute_citation_penalty(result, penalty_weight=2.0)
        assert penalty == 2.0

    def test_penalty_multiple_hallucinations(self):
        """Test penalty scales with number of hallucinations."""
        result = CitationValidationResult(
            is_valid=False,
            cited_docs={"99999", "88888", "77777"},
            valid_docs={"12345"},
            invalid_docs={"99999", "88888", "77777"},
            hallucination_count=3,
        )
        penalty = compute_citation_penalty(result, penalty_weight=2.0)
        assert penalty == 6.0  # 2.0 * 3

    def test_penalty_custom_weight(self):
        """Test penalty with custom weight."""
        result = CitationValidationResult(
            is_valid=False,
            cited_docs={"99999"},
            valid_docs={"12345"},
            invalid_docs={"99999"},
            hallucination_count=1,
        )
        penalty = compute_citation_penalty(result, penalty_weight=5.0)
        assert penalty == 5.0


class TestBatchValidateCitations:
    """Test batch citation validation."""

    def test_batch_all_valid(self):
        """Test batch validation with all valid citations."""
        prompts = [
            "Document 1: Evidence A",
            "Document 2: Evidence B",
        ]
        completions = [
            "CLAIM (Document 1): Claim A",
            "CLAIM (Document 2): Claim B",
        ]

        results = batch_validate_citations(prompts, completions)

        assert len(results) == 2
        assert all(r.is_valid for r in results)

    def test_batch_mixed_validity(self):
        """Test batch validation with mixed valid/invalid."""
        prompts = [
            "Document 1: Evidence A",
            "Document 2: Evidence B",
        ]
        completions = [
            "CLAIM (Document 1): Valid claim",
            "CLAIM (Document 999): Hallucinated claim",
        ]

        results = batch_validate_citations(prompts, completions)

        assert len(results) == 2
        assert results[0].is_valid is True
        assert results[1].is_valid is False
        assert results[1].invalid_docs == {"999"}

    def test_batch_length_mismatch(self):
        """Test that batch validation raises on length mismatch."""
        prompts = ["Document 1: Evidence"]
        completions = ["Claim 1", "Claim 2"]  # Length mismatch

        with pytest.raises(ValueError, match="must have same length"):
            batch_validate_citations(prompts, completions)

    def test_batch_empty(self):
        """Test batch validation with empty lists."""
        results = batch_validate_citations([], [])
        assert results == []


class TestCitationValidationStats:
    """Test aggregate statistics computation."""

    def test_stats_all_valid(self):
        """Test stats when all citations are valid."""
        results = [
            CitationValidationResult(True, set(), set(), set(), 0),
            CitationValidationResult(True, set(), set(), set(), 0),
            CitationValidationResult(True, set(), set(), set(), 0),
        ]

        stats = citation_validation_stats(results)

        assert stats["valid_rate"] == 1.0
        assert stats["mean_hallucinations"] == 0.0
        assert stats["total_hallucinations"] == 0
        assert stats["total_samples"] == 3

    def test_stats_all_invalid(self):
        """Test stats when all citations are invalid."""
        results = [
            CitationValidationResult(False, set(), set(), set(), 1),
            CitationValidationResult(False, set(), set(), set(), 2),
            CitationValidationResult(False, set(), set(), set(), 1),
        ]

        stats = citation_validation_stats(results)

        assert stats["valid_rate"] == 0.0
        assert stats["mean_hallucinations"] == 4.0 / 3  # (1+2+1)/3
        assert stats["total_hallucinations"] == 4
        assert stats["total_samples"] == 3

    def test_stats_mixed(self):
        """Test stats with mixed valid/invalid."""
        results = [
            CitationValidationResult(True, set(), set(), set(), 0),
            CitationValidationResult(False, set(), set(), set(), 2),
            CitationValidationResult(True, set(), set(), set(), 0),
            CitationValidationResult(False, set(), set(), set(), 1),
        ]

        stats = citation_validation_stats(results)

        assert stats["valid_rate"] == 0.5  # 2/4
        assert stats["mean_hallucinations"] == 0.75  # 3/4
        assert stats["total_hallucinations"] == 3
        assert stats["total_samples"] == 4

    def test_stats_empty(self):
        """Test stats with empty results."""
        stats = citation_validation_stats([])

        assert stats["valid_rate"] == 0.0
        assert stats["mean_hallucinations"] == 0.0
        assert stats["total_hallucinations"] == 0
        assert stats["total_samples"] == 0

    def test_stats_scifact_distribution(self):
        """Test stats matching SciFact 50-sample distribution (4% invalid)."""
        # Simulate 50 samples: 2 invalid (4%), 48 valid (96%)
        results = [
            CitationValidationResult(False, set(), set(), set(), 2),  # Claim 133
            CitationValidationResult(False, set(), set(), set(), 1),  # Claim 179
        ] + [
            CitationValidationResult(True, set(), set(), set(), 0) for _ in range(48)
        ]

        stats = citation_validation_stats(results)

        assert stats["valid_rate"] == 0.96
        assert stats["mean_hallucinations"] == 0.06  # 3/50
        assert stats["total_hallucinations"] == 3
        assert stats["total_samples"] == 50
