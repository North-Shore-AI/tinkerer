# Issue: Emergency Fix - Semantic Validation Implementation

**Date:** 2025-11-11
**Status:** ✅ FIXED
**Priority:** CRITICAL
**Component:** CNS 3.0 Proposer Agent Evaluation

## Problem Summary

The CNS Proposer agent evaluation pipeline was using **exact-match metrics** to evaluate LoRA fine-tuned models (rank=8-32, trained on 32-64 examples). This is **fundamentally incompatible** with how LoRA models work:

- LoRA models learn **patterns**, not verbatim text reproduction
- Exact-match on held-out data is logically contradictory for pattern-learning models
- All evaluation metrics showed 0%, hiding actual model performance

This violated **AGENTS.md Section 1.0**, which explicitly retired exact-match in favor of semantic grounding.

## Root Cause

The evaluation logic in `thinker/evaluation.py` (lines 35-42, 68) performed exact string comparisons:

```python
# Line 39: OLD exact-match logic
if pred.strip() == gold.strip():
    matches += 1

# Line 68: OLD c1 exact-match
if c1 and c1.text == claim["claim"]:
    metrics["c1_exact_match"] += 1
```

This approach is incompatible with:
1. LoRA's low-rank adaptation (learns patterns, not sequences)
2. Small training sets (32-64 examples)
3. Paraphrasing and semantic equivalence

## Solution Implemented

### 1. Created 4-Stage Semantic Validation Pipeline

Implemented `thinker/semantic_validation.py` with validation per AGENTS.md Section 4.1:

**Stage 1: Citation Accuracy (Hard Gate)**
- Validates that cited evidence IDs exist in corpus
- Ensures citations support claim polarity
- Binary pass/fail (short-circuit if failed)

**Stage 2: Entailment Score**
- Uses DeBERTa-v3-large NLI model
- Computes entailment probability (evidence → claim)
- Threshold: ≥0.75 (per AGENTS.md Section 1.1)

**Stage 3: Semantic Similarity**
- Uses sentence-transformers (all-MiniLM-L6-v2)
- Computes cosine similarity between generated and gold claims
- Threshold: ≥0.70 (target: ≥60% pass rate)

**Stage 4: Paraphrase Tolerance**
- Accepts valid rephrasings when stages 1-2 pass
- Allows semantic equivalence without exact wording match

### 2. Updated Evaluation Metrics

Modified `thinker/evaluation.py` to track AGENTS.md Section 1.1 compliant metrics:

```python
metrics = {
    "schema_compliance_rate": ...,     # % with CLAIM[c*] structure (target: ≥95%)
    "citation_accuracy_rate": ...,     # % with valid citations (hard gate)
    "mean_entailment_score": ...,      # Avg DeBERTa score (threshold: ≥0.75)
    "entailment_pass_rate": ...,       # % passing entailment threshold
    "mean_semantic_similarity": ...,   # Avg cosine similarity (threshold: ≥0.70)
    "semantic_similarity_rate": ...,   # % passing similarity threshold (target: ≥60%)
    "paraphrase_acceptance_rate": ..., # % passing stage 4
    "overall_pass_rate": ...,          # % passing all 4 stages
    # Legacy metrics (for comparison only)
    "c1_exact_match_rate_LEGACY": ...,
    "evidence_exact_match_avg_LEGACY": ...,
}
```

### 3. Added Dependencies

Updated `requirements.txt`:
```
transformers  # For DeBERTa-v3-NLI
```

(sentence-transformers and torch were already present)

## Comparison Results

Generated `runs/comparison_report.txt` comparing old vs new metrics on 30 evaluation examples:

### Metrics Comparison

| Metric | Old (Exact-Match) | New (Semantic Validation) |
|--------|-------------------|---------------------------|
| C1 Match Rate | 0.0% | N/A (replaced) |
| Schema Compliance | N/A | 0.0% (ISSUE FOUND) |
| Citation Accuracy | N/A | 3.3% (ISSUE FOUND) |
| Mean Entailment | N/A | 0.000 |
| Mean Similarity | N/A | 0.000 |
| **Overall Pass Rate** | **0.0%** | **0.0%** |

## Critical Findings

The semantic validation revealed **deeper issues** beyond just evaluation metrics:

### Finding 1: Schema Compliance Failure (0%)
The model outputs do NOT consistently produce the required `CLAIM[c*]` format. Instead, they produce:
```
Claim: <text>
Relation: <text>
```

**Root Cause:** Training prompts or data formatting don't enforce CLAIM[c*] schema.

**Fix Required:** Update training prompts to include explicit examples of CLAIM[c*] format.

### Finding 2: Citation Accuracy Failure (3.3%)
The model does NOT properly cite evidence documents. Only 1/30 examples correctly referenced source documents.

**Root Cause:** Model not learning to extract and cite Document IDs from prompt.

**Fix Required:** Update training data to include explicit citation examples.

### Finding 3: Diagnostic Value
The semantic validation provided **actionable diagnostic insights** that exact-match could not:
- Exact-match: "0% match" (no diagnostic information)
- Semantic validation: "Schema compliance 0%, citations 3.3%" (specific failure modes identified)

## Next Steps (Recommended)

### 1. Fix Schema Compliance (URGENT)
Update training prompts to enforce CLAIM[c*] format:

```python
# Add to training prompt examples:
"""
CLAIM[c1]: <main claim text>
CLAIM[c2]: <supporting claim from evidence>
RELATION: c2 supports c1
"""
```

### 2. Fix Citation Training (URGENT)
Add explicit citation training examples:

```python
# Training example should show:
"""
Given hypothesis and Document 12345678...
CLAIM[c1]: <claim referencing Document 12345678>
"""
```

### 3. Re-train with Updated Prompts
After fixing prompts:
- Re-run LoRA training with updated prompt format
- Re-evaluate with semantic validation
- Target metrics per AGENTS.md Section 1.1:
  - Schema compliance: ≥95%
  - Citation accuracy: 100% (hard gate)
  - Semantic similarity: ≥60% pass rate

### 4. Monitor Semantic Metrics
Going forward, use semantic validation as the primary evaluation method. Track:
- Schema compliance trend
- Citation accuracy trend
- Entailment score distribution
- Similarity score distribution

## Files Changed

### Created
- `thinker/semantic_validation.py` - 4-stage validation pipeline (368 lines)
- `generate_comparison_report.py` - Comparison report generator (264 lines)
- `ISSUE_semantic_validation_emergency_fix.md` - This document

### Modified
- `thinker/evaluation.py` - Integrated semantic validation (208 lines changed)
- `requirements.txt` - Added transformers dependency

### Generated
- `runs/comparison_report.txt` - Detailed comparison of old vs new metrics

## Verification

To verify the fix:

```bash
# Run evaluation with new semantic validation
cd /home/home/p/g/North-Shore-AI/tinkerer
source .venv/bin/activate
python3 -m thinker.evaluation

# Should see new metrics output:
# ================================================================================
# 4-STAGE SEMANTIC VALIDATION METRICS (AGENTS.md Section 1.1)
# ================================================================================
# Schema Compliance:     X.X%
# Citation Accuracy:     X.X%
# Mean Entailment Score: X.XXX
# ...
```

## Alignment with AGENTS.md

This fix ensures compliance with:

- **Section 1.0 "Evaluation Philosophy (Exact-Match Exit)"**: Exact-match is now legacy metric only
- **Section 1.1 "Proposer Health Metrics"**: All specified metrics now tracked
- **Section 4.1 "Semantic Grounding: Operational Definition"**: 4-stage validation implemented exactly as specified

## Lessons Learned

1. **Evaluation metrics must match model architecture**: LoRA pattern-learning requires semantic evaluation, not exact-match
2. **Semantic validation provides better diagnostics**: Identified specific failure modes (schema, citations) that exact-match hid
3. **Follow specification**: AGENTS.md Section 1.0 explicitly warned against exact-match, but it was used anyway
4. **Test assumptions early**: Running semantic validation earlier would have caught the schema/citation issues sooner

## References

- AGENTS.md Section 1.0: Evaluation Philosophy (Exact-Match Exit)
- AGENTS.md Section 1.1: Proposer Health Metrics
- AGENTS.md Section 4.1: Semantic Grounding: Operational Definition
- runs/comparison_report.txt: Full comparison results
- thinker/semantic_validation.py: Implementation

---

**Resolution Status:** ✅ FIXED (semantic validation implemented)
**Blocker Status:** ⚠️ BLOCKED on schema compliance and citation training fixes (recommended next steps)
