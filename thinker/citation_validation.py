"""Citation validation utilities for training and evaluation.

This module provides functions to detect citation hallucination - when models
cite documents that don't exist in the source corpus.

Per HIGH_SEVERITY_REVIEW.md (2025-11-18), citation hallucination is a critical
failure mode that must be detected and penalized during training.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Set, Tuple


@dataclass
class CitationValidationResult:
    """Result of citation validation."""

    is_valid: bool
    cited_docs: Set[str]
    valid_docs: Set[str]
    invalid_docs: Set[str]
    hallucination_count: int


def extract_document_ids(text: str) -> Set[str]:
    """Extract document IDs from text.

    Looks for patterns like:
    - "Document 12345:"
    - "Document 12345678:"
    - "(Document 12345)"

    Args:
        text: Text to extract document IDs from (prompt or completion)

    Returns:
        Set of document ID strings (e.g., {"12345", "67890"})

    Example:
        >>> extract_document_ids("Document 12345: foo\\nDocument 67890: bar")
        {'12345', '67890'}
    """
    # Match various document ID patterns
    patterns = [
        r"Document\s+(\d+):",  # Document 12345:
        r"\(Document\s+(\d+)\)",  # (Document 12345)
        r"CLAIM\[c\d+\]\s+\(Document\s+(\d+)\)",  # CLAIM[c1] (Document 12345)
    ]

    doc_ids: Set[str] = set()
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            doc_ids.add(match.group(1))

    return doc_ids


def validate_citations(prompt: str, completion: str) -> CitationValidationResult:
    """Validate that all cited documents exist in the prompt.

    Args:
        prompt: The input prompt containing available documents
        completion: The model's completion that may cite documents

    Returns:
        CitationValidationResult with validation details

    Example:
        >>> prompt = "Document 12345: Evidence about X"
        >>> completion = "CLAIM[c1] (Document 12345): X is true"
        >>> result = validate_citations(prompt, completion)
        >>> result.is_valid
        True
        >>> completion_bad = "CLAIM[c1] (Document 99999): X is true"
        >>> result = validate_citations(prompt, completion_bad)
        >>> result.is_valid
        False
        >>> result.invalid_docs
        {'99999'}
    """
    valid_docs = extract_document_ids(prompt)
    cited_docs = extract_document_ids(completion)

    invalid_docs = cited_docs - valid_docs
    is_valid = len(invalid_docs) == 0

    return CitationValidationResult(
        is_valid=is_valid,
        cited_docs=cited_docs,
        valid_docs=valid_docs,
        invalid_docs=invalid_docs,
        hallucination_count=len(invalid_docs),
    )


def compute_citation_penalty(
    result: CitationValidationResult, penalty_weight: float = 2.0
) -> float:
    """Compute citation hallucination penalty for training loss.

    Args:
        result: CitationValidationResult from validate_citations()
        penalty_weight: Weight multiplier for hallucination penalty (from config)

    Returns:
        Penalty value to add to training loss

    Example:
        >>> result = CitationValidationResult(
        ...     is_valid=False,
        ...     cited_docs={"12345", "99999"},
        ...     valid_docs={"12345"},
        ...     invalid_docs={"99999"},
        ...     hallucination_count=1
        ... )
        >>> compute_citation_penalty(result, penalty_weight=2.0)
        2.0
    """
    if result.is_valid:
        return 0.0

    # Penalty is proportional to number of hallucinated citations
    return penalty_weight * result.hallucination_count


def batch_validate_citations(
    prompts: list[str], completions: list[str]
) -> list[CitationValidationResult]:
    """Validate citations for a batch of prompt-completion pairs.

    Args:
        prompts: List of input prompts
        completions: List of model completions

    Returns:
        List of CitationValidationResult, one per pair

    Example:
        >>> prompts = ["Document 1: foo", "Document 2: bar"]
        >>> completions = ["CLAIM (Document 1): X", "CLAIM (Document 99): Y"]
        >>> results = batch_validate_citations(prompts, completions)
        >>> results[0].is_valid
        True
        >>> results[1].is_valid
        False
    """
    if len(prompts) != len(completions):
        raise ValueError(
            f"Prompts and completions must have same length: {len(prompts)} vs {len(completions)}"
        )

    return [validate_citations(p, c) for p, c in zip(prompts, completions)]


def citation_validation_stats(results: list[CitationValidationResult]) -> dict[str, float]:
    """Compute aggregate statistics for citation validation.

    Args:
        results: List of CitationValidationResult from batch validation

    Returns:
        Dictionary with statistics:
        - valid_rate: Fraction of samples with valid citations
        - mean_hallucinations: Average number of hallucinated citations per sample
        - total_hallucinations: Total count of hallucinated citations
        - total_samples: Number of samples

    Example:
        >>> results = [
        ...     CitationValidationResult(True, set(), set(), set(), 0),
        ...     CitationValidationResult(False, set(), set(), set(), 2),
        ... ]
        >>> stats = citation_validation_stats(results)
        >>> stats['valid_rate']
        0.5
        >>> stats['mean_hallucinations']
        1.0
    """
    if not results:
        return {
            "valid_rate": 0.0,
            "mean_hallucinations": 0.0,
            "total_hallucinations": 0,
            "total_samples": 0,
        }

    valid_count = sum(1 for r in results if r.is_valid)
    total_hallucinations = sum(r.hallucination_count for r in results)

    return {
        "valid_rate": valid_count / len(results),
        "mean_hallucinations": total_hallucinations / len(results),
        "total_hallucinations": total_hallucinations,
        "total_samples": len(results),
    }
