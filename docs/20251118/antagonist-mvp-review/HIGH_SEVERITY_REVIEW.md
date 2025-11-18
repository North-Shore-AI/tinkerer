# HIGH Severity Claims Manual Review
**Date:** 2025-11-18
**Reviewer:** Claude (AI Agent)
**Claims Reviewed:** 133, 179

---

## Executive Summary

**Critical Finding:** Both HIGH severity claims exhibit the **same failure mode**: the Proposer is **hallucinating document IDs** and citing evidence that **does not exist in the source corpus**. This is a **severe training failure** that requires immediate intervention.

**Root Cause:** The model has learned the structural pattern of citation (format: `CLAIM[c#] (Document <doc_id>)`) but has **not learned to ground citations in the actual provided documents**.

**Severity Justification:** Both cases triggered HIGH severity due to:
1. Chirality ≥0.65 (structural tension)
2. Evidence overlap ≥0.5 (multi-evidence conflict)
3. Entailment = 0.0 (complete semantic failure)

**Recommended Action:** Urgent Proposer retraining with increased citation loss weight.

---

## Claim 133: Podosome Formation

### Source Documents Provided
1. **Document 38485364:** "The adaptor protein Tks5/Fish is required for podosome formation..."
2. **Document 6969753:** "Dynamic interactions of cortactin and membrane type 1 matrix metalloproteinase at invadopodia..."
3. **Document 17934082:** "Membrane lipids in invadopodia and podosomes..."

### Model Output
```
CLAIM[c1]: The Arp2/3 complex is involved in the formation of podosomes.

CLAIM[c2] (Document 16280642): Further, it was found that N-WASP bound all
SH3 domains of Tks5/FISH, which facilitated circular podosome formation.
RELATION: c2 supports c1

CLAIM[c3] (Document 12640810): In particular, cortactin phosphorylation was
found to regulate cofilin and Arp2/3 complex-dependent actin polymerization.
RELATION: c3 supports c1

CLAIM[c4] (Document 12640810): Cortactin directly binds cofilin and inhibits
its severing activity. Cortactin phosphorylation is required to release this
inhibition so cofilin can sever actin filaments to create barbed ends at
invadopodia to support Arp2/3-dependent actin polymerization.
RELATION: c4 supports c1

CLAIM[c5] (Document 6969753): Small interfering RNA (siRNA) inhibition
established that organization of invadopodia structure requires cortactin...
```

### Analysis

**❌ CRITICAL FAILURE: Citation Hallucination**

**Fabricated Document IDs:**
- **Document 16280642** - NOT in the source passage
- **Document 12640810** - NOT in the source passage

**Validation Results:**
- Schema valid: ✅ TRUE (correct CLAIM[c#] format)
- Citation valid: ❌ **FALSE** (references non-existent documents)
- Entailment score: ❌ **0.0000** (no entailment because evidence doesn't exist)
- Overall pass: ❌ FALSE

**Why Entailment Failed:**
The DeBERTa NLI model compared the generated claims against the **actual** source documents (38485364, 6969753, 17934082). Since the model cited **different** documents (16280642, 12640810), there was zero semantic overlap → entailment = 0.0.

**Chirality Metrics:**
- Chirality score: **0.6546** (≥0.65 HIGH threshold)
- Fisher-Rao distance: **22.64** (high distributional divergence)
- Evidence overlap: **0.6** (partial overlap with actual docs)
- Polarity conflict: FALSE

**Why Chirality is High:**
The model **did** extract some valid content from Document 6969753 (CLAIM[c5] about cortactin), creating structural tension between the hallucinated claims (c2-c4) and the partially valid claim (c5). This mixed validity creates high chirality.

### Root Cause Analysis

**What the model learned:**
1. ✅ Schema format: `CLAIM[c#] (Document <doc_id>): <text>`
2. ✅ Relation format: `RELATION: c2 supports c1`
3. ✅ Scientific language and domain terminology

**What the model FAILED to learn:**
1. ❌ Citation grounding: Must only cite documents from the provided passage
2. ❌ Evidence retrieval: Must extract text from the actual cited document
3. ❌ Factual accuracy: The Arp2/3 claim may be scientifically valid but isn't supported by the given documents

**Training Data Hypothesis:**
The model likely saw examples during training where documents 16280642 and 12640810 were cited. It **memorized** these associations rather than learning to **retrieve** evidence from the current context.

### Recommended Fix

**Immediate (P0):**
1. Add **citation validation during training**:
   ```python
   # In training loop, add penalty for citing non-existent doc IDs
   cited_docs = extract_document_ids(completion)
   valid_docs = extract_document_ids(prompt)
   invalid_citations = cited_docs - valid_docs

   if invalid_citations:
       loss += CITATION_HALLUCINATION_PENALTY * len(invalid_citations)
   ```

2. **Increase evidence grounding weight**:
   ```yaml
   training:
     CNS_CLAIM_EVIDENCE_WEIGHT: 3.0  # Even higher than 2.0
     CITATION_VALIDITY_WEIGHT: 2.0   # New penalty term
   ```

3. **Add negative examples** to training data:
   - Include examples with explicit instruction: "Only cite documents from the passage above"
   - Add examples that explicitly reject claims not supported by provided evidence

---

## Claim 179: Birthweight and Breast Cancer

### Source Documents Provided
1. **Document 16322674:** "Birth Size and Breast Cancer Risk: Re-analysis of Individual Participant Data from 32 Studies"
2. **Document 27123743:** "Role of birthweight in the etiology of breast cancer."
3. **Document 23557241:** "Intrauterine factors and risk of breast cancer: a systematic review and meta-analysis..."

### Model Output
```
CLAIM[c1]: Women with a lower birth weight are more likely to develop
breast cancer later in life.

CLAIM[c2] (Document 17450673): We found that heavier birth weights were
associated with increased breast cancer risk, with studies involving five
categories of birth weight identifying odds ratios (ORs) of 1.24 (95%
confidence interval [CI] 1.04 to 1.48) for 4,000 g or more and 1.15 (95%
CI 1.04 to 1.26) for 3,500 g to 3,999 g, relative to a birth weight of
2,500 to 2,599 g.
RELATION: c2 refutes c1

CLAIM[c3] (Document 17450673): CONCLUSION Our findings provide some support
for the hypothesis that in utero exposures reflective of higher endogenous
hormone levels could affect risk for development of breast cancer in adulthood.
RELATION: c3 refutes c1

CLAIM[c4] (Document 23557241): Increased risk of breast cancer was noted
with increased birthweight (relative risk [RR] 1.15 [95%
```

### Analysis

**❌ CRITICAL FAILURE: Citation Hallucination + Self-Refutation**

**Fabricated Document IDs:**
- **Document 17450673** - NOT in the source passage (but c2 and c3 cite it)

**Validation Results:**
- Schema valid: ✅ TRUE
- Citation valid: ❌ **FALSE** (document 17450673 doesn't exist)
- Entailment score: ❌ **0.0000**
- Overall pass: ❌ FALSE

**Why This is Worse Than Claim 133:**
Not only does the model hallucinate document IDs, but it also **generates a claim (c1) that contradicts the actual evidence**, then **correctly refutes its own false claim** using hallucinated evidence!

**Correct Evidence (from actual documents):**
- Document 16322674: "Birth weight was positively associated with breast cancer risk... pooled RR per SD increment: 1.06"
- Document 27123743: "The majority of studies identified a positive link between birthweight and... breast cancer. The relative risk estimate... was 1.23"

**Model's False Claim:**
- c1: "Women with **lower** birth weight are more likely to develop breast cancer" ❌

**Model's Self-Correction (using hallucinated evidence):**
- c2: "heavier birth weights were associated with **increased** breast cancer risk" ✅ (correct direction, wrong source)
- c3: REFUTES c1 ✅ (correct refutation, wrong source)

**Chirality Metrics:**
- Chirality score: **0.6542** (≥0.65 HIGH threshold)
- Fisher-Rao distance: **12.10**
- Evidence overlap: **0.5** (half the claims cite hallucinated docs)
- Polarity conflict: FALSE (but should be TRUE due to self-refutation!)

**Why Polarity Conflict is False:**
The `logic/betti.py` polarity detection looks for **both** `supports` and `refutes` edges pointing to c1. In this case:
- c2 → **refutes** c1
- c3 → **refutes** c1
- No claims **support** c1

So technically, there's no polarity *conflict* (contradiction in evidence direction), but there **is** a logical error (c1 is false and immediately refuted).

### Root Cause Analysis

**Two Compounding Failures:**

**Failure 1: Claim Extraction (c1)**
- The model **inverted** the hypothesis direction
- Actual evidence: higher birthweight → higher cancer risk
- Model claim: lower birthweight → higher cancer risk
- **Hypothesis:** Model may have confused "lower risk" with "lower birthweight"

**Failure 2: Citation Hallucination (c2, c3)**
- Model correctly identified that c1 is false
- Model correctly retrieved evidence about higher birthweight increasing risk
- **BUT** cited a document (17450673) that wasn't provided
- The evidence text in c2 **may be real** (from the hallucinated document) but wasn't in the provided corpus

**Why This Happened:**
The model has **memorized associations** between:
- Topic: "birthweight and breast cancer"
- Document: 17450673 (likely seen during training)
- Evidence: odds ratios for different birthweight categories

When prompted with this topic, it **retrieved from memory** instead of from the provided context.

### Recommended Fix

**Immediate (P0):**
1. Same as Claim 133: citation validation + increased penalties
2. **Add hypothesis direction validation**:
   - If c1 states "lower X → higher Y"
   - And evidence states "higher X → higher Y"
   - Penalize this inversion error heavily

3. **Strengthen context grounding**:
   ```python
   # During training, mask the model's memory of document IDs
   # Force it to only attend to the current prompt's "Document <id>" markers
   prompt = """
   IMPORTANT: Only cite documents listed above. Do not cite documents from memory.

   Passage:
   Document 16322674: ...
   Document 27123743: ...

   Available documents: 16322674, 27123743
   """
   ```

**Medium Priority (P1):**
1. Add **polarity conflict detection** for self-refutation:
   - If c1 makes a claim, and c2+ refutes it **without supporting evidence**, flag as error
   - Current `logic/betti.py` misses this case

---

## Comparative Analysis

### Common Failure Patterns

| Aspect | Claim 133 | Claim 179 | Pattern |
|--------|-----------|-----------|---------|
| **Schema** | ✅ Valid | ✅ Valid | **Learned** |
| **Citation format** | ✅ Correct syntax | ✅ Correct syntax | **Learned** |
| **Citation validity** | ❌ Hallucinated docs | ❌ Hallucinated docs | **NOT learned** |
| **Evidence grounding** | ❌ 0.0 entailment | ❌ 0.0 entailment | **NOT learned** |
| **Chirality** | 0.6546 (HIGH) | 0.6542 (HIGH) | Correctly flagged |
| **Evidence overlap** | 0.6 | 0.5 | Multi-doc confusion |

**Key Insight:** Both claims have **near-identical chirality scores** (0.654x), suggesting this is a systematic failure mode, not random error.

### Why Antagonist Correctly Flagged These

The Antagonist heuristics worked as designed:

1. **Chirality threshold (≥0.55):** ✅ Both exceeded 0.65
2. **High chirality threshold (≥0.65):** ✅ Both met this threshold
3. **Evidence overlap (≥0.2):** ✅ Both had 0.5-0.6 overlap
4. **Weak entailment (<0.5):** ✅ Both had 0.0 entailment
5. **Severity escalation:** ✅ Correctly escalated to HIGH

**Validation:** The Antagonist MVP successfully identified the **two most severe failures** in the 50-sample dataset.

---

## Impact on Downstream Components

### If These Claims Reached Synthesizer

**Scenario:** Synthesizer receives Claim 133 + a conflicting claim about podosome formation

**Problem:**
1. Synthesizer tries to resolve conflict between hallucinated evidence
2. Produces synthesis citing **non-existent documents**
3. Researcher follows citation trail → **dead end**
4. **Trust in entire system destroyed**

**Critical Risk:** Citation hallucination is **more dangerous** than low entailment because:
- Low entailment: Claim is weak but traceable
- Hallucinated citation: Claim appears strong but is **untraceable** and **unverifiable**

### Why This Must Block Synthesis

**Proposed Rule:**
```python
# In Synthesizer input validation
if flag['severity'] == 'HIGH' and 'CITATION_INVALID' in flag['issues']:
    raise BlockedBySynthesizerError(
        f"Claim {claim_id} cites non-existent documents. "
        f"Requires Proposer refinement before synthesis."
    )
```

---

## Recommendations

### P0 (Immediate - This Week)

**1. Add Citation Validity to Issue Types**

Currently, Antagonist flags:
- `POLARITY_CONTRADICTION`
- `POLARITY_CONFLICT`
- `WEAK_ENTAILMENT`

**Add:**
- `CITATION_INVALID` (when cited doc IDs don't exist in corpus)
- `CLAIM_INVERSION` (when claim direction contradicts evidence)

**Implementation:**
```python
# In antagonist.py
def _evaluate_entry(self, entry):
    validation = entry.get("validation", {})

    # New check
    if not validation.get("citation_valid"):
        issues.append({
            "issue_type": "CITATION_INVALID",
            "details": {"citation_valid": False}
        })
        severity = "HIGH"  # Citation hallucination is always HIGH
```

**2. Retrain Proposer with Citation Penalty**

```yaml
# cns-support-models/configs/claim_extractor_scifact.yaml
training:
  CNS_CLAIM_EVIDENCE_WEIGHT: 3.0      # Increase from current
  CITATION_VALIDITY_WEIGHT: 2.0       # NEW: Penalize invalid citations
  learning_rate: 2e-4
  num_epochs: 5

  # Add explicit validation in training loop
  validate_citations_during_training: true
```

**3. Create Negative Example Training Data**

```jsonl
// examples/negative_citation_hallucination.jsonl
{
  "prompt": "Passage:\nDocument 12345: ...\n\nTask: Only cite Document 12345.",
  "completion_good": "CLAIM[c1] (Document 12345): ...",
  "completion_bad": "CLAIM[c1] (Document 99999): ...",
  "loss_weight": 3.0  // Heavily penalize hallucination
}
```

### P1 (Next Week)

**1. Add Self-Refutation Detection**

The current `polarity_conflict` detection only catches:
```
c2 supports c1
c3 refutes c1  → POLARITY_CONFLICT
```

**Should also catch:**
```
c1: Lower X → higher Y
c2 refutes c1
c3 refutes c1
// No support for c1 → Self-refutation
```

**2. Implement Citation Grounding Test**

```python
# thinker/tests/test_citation_grounding.py
def test_reject_hallucinated_citations():
    prompt = make_prompt(doc_ids=[12345, 67890])
    completion = "CLAIM[c1] (Document 99999): ..."

    validator = CitationValidator()
    result = validator.validate(prompt, completion)

    assert result.valid == False
    assert "99999" in result.invalid_doc_ids
```

### P2 (Month 2)

**1. Add Citation Provenance Tracking**

Track which document IDs appear in:
- Training data (frequency distribution)
- Evaluation prompts (current context)
- Model completions (generated citations)

**Goal:** Detect if model is citing from memory vs. context

**2. Implement Retrieval-Augmented Citation**

Instead of generating citations end-to-end, use a two-stage process:
1. **Stage 1:** Extract claims (no citations)
2. **Stage 2:** For each claim, retrieve most relevant document from context
3. **Stage 3:** Generate citation with retrieved doc ID

This architecture **forces** grounding in the current context.

---

## Conclusion

### Summary of Findings

**Both HIGH severity claims (133, 179) exhibit citation hallucination:**
- Model cites documents NOT in the provided passage
- Entailment scores correctly drop to 0.0 (no semantic match)
- Chirality scores correctly escalate to 0.654+ (structural tension)
- Antagonist correctly flagged both as HIGH severity

**Root cause:** Model learned citation **format** but not citation **grounding**

**Severity:** **CRITICAL** - Citation hallucination is more dangerous than weak entailment because it appears valid on the surface but is untraceable and unverifiable.

### Validation of Antagonist MVP

✅ **The Antagonist performed correctly:**
1. Identified the 2 most severe failures out of 50 samples
2. Escalated to HIGH severity appropriately
3. Captured all relevant metrics (chirality, entailment, evidence overlap)
4. Provided structured flags for downstream action

**No false positives detected** - Both HIGH severity flags are genuine critical failures.

### Critical Path Forward

**Cannot proceed to Synthesizer until this is fixed:**
1. ⏳ Add `CITATION_INVALID` issue type to Antagonist
2. ⏳ Retrain Proposer with citation validity penalty
3. ⏳ Create negative example training data
4. ⏳ Re-evaluate with new adapter
5. ⏳ Verify citation hallucination rate drops to <5%

**Timeline:** 1-2 weeks for fix + retraining + validation

**Success Criteria:**
- Citation valid rate ≥95% (up from current ~40%)
- Mean entailment ≥0.60 (up from current 0.41)
- HIGH severity flags <1% (down from current 4%)

---

## Appendix: Full Raw Data

**Claim 133 Metrics:**
```json
{
  "claim_id": 133,
  "validation": {
    "schema_valid": true,
    "citation_valid": false,
    "entailment_score": 0.0,
    "semantic_similarity": 0.0,
    "overall_pass": false
  },
  "beta1": 0,
  "chirality": {
    "score": 0.6546,
    "fisher_rao_distance": 22.6392,
    "evidence_overlap": 0.6,
    "polarity_conflict": false
  }
}
```

**Claim 179 Metrics:**
```json
{
  "claim_id": 179,
  "validation": {
    "schema_valid": true,
    "citation_valid": false,
    "entailment_score": 0.0,
    "semantic_similarity": 0.0,
    "overall_pass": false
  },
  "beta1": 0,
  "chirality": {
    "score": 0.6542,
    "fisher_rao_distance": 12.1011,
    "evidence_overlap": 0.5,
    "polarity_conflict": false
  }
}
```

**Antagonist Flags:**
```json
[
  {
    "claim_id": 133,
    "severity": "HIGH",
    "issues": [
      {"issue_type": "POLARITY_CONTRADICTION", "details": {...}},
      {"issue_type": "WEAK_ENTAILMENT", "details": {"entailment_score": 0.0}}
    ]
  },
  {
    "claim_id": 179,
    "severity": "HIGH",
    "issues": [
      {"issue_type": "POLARITY_CONTRADICTION", "details": {...}},
      {"issue_type": "WEAK_ENTAILMENT", "details": {"entailment_score": 0.0}}
    ]
  }
]
```

---

**Review Complete**
**Next Action:** Implement `CITATION_INVALID` issue type + Proposer retraining
