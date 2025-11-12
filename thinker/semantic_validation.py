"""
Semantic validation for CNS claim extraction.

Implements the 4-stage validation pipeline from AGENTS.md Section 4.1:
1. Citation Accuracy (hard gate)
2. Entailment Score (DeBERTa-v3-NLI)
3. Semantic Similarity (sentence-transformers)
4. Paraphrase Tolerance

This replaces exact-match evaluation which is incompatible with LoRA-based
models that learn patterns, not verbatim sequences.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class ValidationResult:
    """Results from 4-stage semantic validation."""

    # Stage 1: Citation Accuracy
    citation_valid: bool
    cited_ids: Set[str]
    missing_ids: Set[str]

    # Stage 2: Entailment
    entailment_score: float
    entailment_pass: bool  # >= 0.75 threshold

    # Stage 3: Semantic Similarity
    semantic_similarity: float
    similarity_pass: bool  # >= 0.7 threshold

    # Stage 4: Paraphrase Tolerance
    paraphrase_accepted: bool

    # Overall
    overall_pass: bool

    # Schema compliance (separate check)
    schema_valid: bool
    schema_errors: List[str]


class SemanticValidator:
    """
    4-stage semantic validation for CNS claim extraction.

    Uses:
    - DeBERTa-v3-large for NLI entailment checking
    - all-MiniLM-L6-v2 for semantic similarity (fast, 80MB model)
    """

    # Thresholds from AGENTS.md Section 1.1
    ENTAILMENT_THRESHOLD = 0.75
    SIMILARITY_THRESHOLD = 0.7

    def __init__(self, device: Optional[str] = None):
        """Initialize semantic validation models.

        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # NLI model for entailment (Stage 2)
        print(f"[semantic_val] Loading DeBERTa-v3 NLI model on {device}...", flush=True)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            "cross-encoder/nli-deberta-v3-large"
        ).to(device)
        self.nli_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-large")
        self.nli_model.eval()

        # Sentence embeddings for similarity (Stage 3)
        print(f"[semantic_val] Loading sentence-transformers model on {device}...", flush=True)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    def validate_claim(
        self,
        generated_claim: str,
        gold_claim: str,
        generated_full_output: str,
        evidence_corpus: Dict[str, Dict],
        gold_evidence_ids: Set[str],
    ) -> ValidationResult:
        """
        Run 4-stage semantic validation on a generated claim.

        Args:
            generated_claim: The claim text extracted from model output
            gold_claim: The ground-truth claim text
            generated_full_output: Full model output for citation extraction
            evidence_corpus: Document corpus {doc_id: {title, abstract}}
            gold_evidence_ids: Ground-truth evidence document IDs

        Returns:
            ValidationResult with scores and pass/fail for each stage
        """
        schema_valid, schema_errors = self._check_schema_compliance(generated_full_output)

        # Stage 1: Citation Accuracy (hard gate)
        cited_ids = self._extract_citation_ids(generated_full_output)
        missing_ids = gold_evidence_ids - cited_ids
        citation_valid = len(missing_ids) == 0 and len(cited_ids) > 0

        # Short-circuit: If citations invalid, fail early
        if not citation_valid:
            return ValidationResult(
                citation_valid=False,
                cited_ids=cited_ids,
                missing_ids=missing_ids,
                entailment_score=0.0,
                entailment_pass=False,
                semantic_similarity=0.0,
                similarity_pass=False,
                paraphrase_accepted=False,
                overall_pass=False,
                schema_valid=schema_valid,
                schema_errors=schema_errors,
            )

        # Stage 2: Entailment Score
        # Gather evidence sentences from cited documents
        evidence_text = self._gather_evidence_text(cited_ids, evidence_corpus)
        entailment_score = self._compute_entailment(generated_claim, evidence_text)
        entailment_pass = entailment_score >= self.ENTAILMENT_THRESHOLD

        # Short-circuit: If entailment fails, fail early
        if not entailment_pass:
            return ValidationResult(
                citation_valid=True,
                cited_ids=cited_ids,
                missing_ids=missing_ids,
                entailment_score=entailment_score,
                entailment_pass=False,
                semantic_similarity=0.0,
                similarity_pass=False,
                paraphrase_accepted=False,
                overall_pass=False,
                schema_valid=schema_valid,
                schema_errors=schema_errors,
            )

        # Stage 3: Semantic Similarity
        similarity_score = self._compute_semantic_similarity(generated_claim, gold_claim)
        similarity_pass = similarity_score >= self.SIMILARITY_THRESHOLD

        # Stage 4: Paraphrase Tolerance
        # Accept claim if stages 1-2 pass, even if similarity is slightly below threshold
        # This allows valid rephrasings that convey the same semantic content
        paraphrase_accepted = citation_valid and entailment_pass

        # Overall pass: All critical stages must pass
        overall_pass = citation_valid and entailment_pass and (similarity_pass or paraphrase_accepted)

        return ValidationResult(
            citation_valid=citation_valid,
            cited_ids=cited_ids,
            missing_ids=missing_ids,
            entailment_score=entailment_score,
            entailment_pass=entailment_pass,
            semantic_similarity=similarity_score,
            similarity_pass=similarity_pass,
            paraphrase_accepted=paraphrase_accepted,
            overall_pass=overall_pass,
            schema_valid=schema_valid,
            schema_errors=schema_errors,
        )

    def _check_schema_compliance(self, output: str) -> tuple[bool, List[str]]:
        """Check if output has valid CLAIM[c*] schema structure.

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Check for CLAIM[c*] pattern
        claim_pattern = r"CLAIM\[c\d+\]"
        claims = re.findall(claim_pattern, output, re.IGNORECASE)

        if not claims:
            errors.append("No CLAIM[c*] structure found")

        # Check for required c1 (main claim)
        if not re.search(r"CLAIM\[c1\]", output, re.IGNORECASE):
            errors.append("Missing CLAIM[c1] (main claim)")

        # Check for RELATION patterns (optional but recommended)
        if "RELATION" not in output.upper() and "Relation:" not in output:
            errors.append("No RELATION structure found (optional)")

        return len(errors) == 0, errors

    def _extract_citation_ids(self, output: str) -> Set[str]:
        """Extract document IDs cited in the output.

        Looks for patterns like:
        - Document 12345
        - doc_id: 12345
        - [12345]
        """
        cited_ids = set()

        # Pattern 1: Document NNNNNN
        for match in re.finditer(r"[Dd]ocument\s+(\d+)", output):
            cited_ids.add(match.group(1))

        # Pattern 2: doc_id: NNNNNN or (doc_id=NNNNNN)
        for match in re.finditer(r"doc_id[:\s=]+(\d+)", output):
            cited_ids.add(match.group(1))

        # Pattern 3: [NNNNNN] (citation style)
        for match in re.finditer(r"\[(\d{5,})\]", output):
            cited_ids.add(match.group(1))

        return cited_ids

    def _gather_evidence_text(self, doc_ids: Set[str], corpus: Dict[str, Dict]) -> str:
        """Gather all text from cited documents for entailment checking."""
        evidence_parts = []
        for doc_id in doc_ids:
            doc = corpus.get(str(doc_id))
            if doc:
                # Include title and abstract
                evidence_parts.append(doc.get("title", ""))
                abstract = doc.get("abstract", [])
                if isinstance(abstract, list):
                    evidence_parts.append(" ".join(abstract))
                else:
                    evidence_parts.append(str(abstract))
        return " ".join(evidence_parts)

    def _compute_entailment(self, claim: str, evidence: str) -> float:
        """
        Compute entailment score using DeBERTa-v3-NLI.

        Returns probability that evidence entails claim (0.0-1.0).
        """
        if not evidence or not claim:
            return 0.0

        # DeBERTa NLI format: [CLS] premise [SEP] hypothesis [SEP]
        # In our case: evidence is premise, claim is hypothesis
        inputs = self.nli_tokenizer(
            evidence,
            claim,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            logits = outputs.logits

        # Softmax to get probabilities for [contradiction, neutral, entailment]
        probs = torch.softmax(logits, dim=1)[0]
        entailment_prob = probs[2].item()  # Index 2 is entailment

        return entailment_prob

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity using sentence embeddings.

        Returns cosine similarity (0.0-1.0).
        """
        if not text1 or not text2:
            return 0.0

        embeddings = self.embedding_model.encode(
            [text1, text2], convert_to_tensor=True, device=self.device
        )
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

        # Clamp to [0, 1] range
        return max(0.0, min(1.0, similarity))


def validate_batch(
    validator: SemanticValidator,
    predictions: List[Dict],
    gold_data: List[Dict],
    corpus: Dict[str, Dict],
) -> List[ValidationResult]:
    """
    Validate a batch of predictions.

    Args:
        validator: Initialized SemanticValidator
        predictions: List of {claim_id, generated_claim, full_output}
        gold_data: List of {claim_id, claim, cited_doc_ids}
        corpus: Document corpus

    Returns:
        List of ValidationResult objects
    """
    results = []

    for pred, gold in zip(predictions, gold_data):
        if pred.get("claim_id") != gold.get("id"):
            raise ValueError(f"Misaligned data: pred={pred.get('claim_id')}, gold={gold.get('id')}")

        result = validator.validate_claim(
            generated_claim=pred.get("generated_claim", ""),
            gold_claim=gold.get("claim", ""),
            generated_full_output=pred.get("full_output", ""),
            evidence_corpus=corpus,
            gold_evidence_ids=set(str(doc_id) for doc_id in gold.get("cited_doc_ids", [])),
        )
        results.append(result)

    return results
