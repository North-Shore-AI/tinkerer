# Training Ready - Citation Validity Weight 5.0
**Date:** 2025-11-18
**Status:** ✅ READY TO RUN
**Commit:** 34334bd (docs update) on top of e500bb2 (weight=5.0 code change)

---

## Summary

All documentation has been updated to reflect the Antagonist MVP completion and the failed training iteration with `citation_validity_weight=2.0`. The code is configured with the new weight of 5.0 and is ready for the next training run.

---

## What Was Accomplished Today

### ✅ Antagonist MVP Completion
- **Status:** Production-ready MVP shipped
- **Performance:** 92% flagging rate (46/50 samples)
- **Critical Success:** Correctly identified 2 HIGH severity citation hallucinations (claims 133, 179)
- **Test Coverage:** 22 tests passing
- **Documentation:** Complete review in `docs/20251118/antagonist-mvp-review/`

### ✅ Training Iteration Analysis
- **Previous Training (weight=2.0):** FAILED
  - Adapter: `claim-extractor-scifact-20251118T220454`
  - Training: 98.7% loss reduction (successful training metrics)
  - Evaluation: Schema 98% (-2%), Citation 96%, Entailment 0.395 (-0.053, WORSE), Pass 34% (-4%, WORSE)
  - **Critical Finding:** 2 HIGH severity CITATION_INVALID flags persist
  - **Root Cause:** Penalty weight=2.0 (3x loss multiplier) insufficient

### ✅ Code Configuration Update
- **File:** `cns-support-models/scripts/train_claim_extractor.py:41`
- **Change:** `CITATION_VALIDITY_WEIGHT` default 2.0 → 5.0
- **Effect:** 3x loss multiplier → 6x loss multiplier
- **Commit:** e500bb2

### ✅ Documentation Updates
- **README.md:** Updated with training history, Antagonist MVP status, current metrics
- **AGENTS.md:** Added performance history, critical findings, active remediation plans
- **ROADMAP.md:** Updated with completed/active/blocked components, P0/P1/P2 priorities
- **Commit:** 34334bd (263 insertions, 103 deletions)

---

## Current Configuration

### Training Parameters
```python
CLAIM_C1_WEIGHT = 5.0                    # Weight for main hypothesis (c1)
CLAIM_EVIDENCE_WEIGHT = 2.0              # Weight for evidence claims
CITATION_VALIDITY_WEIGHT = 5.0           # ⬆️ INCREASED from 2.0 (6x loss multiplier)
```

### Training Setup
- **Dataset:** SciFact (505 examples)
- **Base Model:** meta-llama/Llama-3.1-8B-Instruct
- **LoRA Config:** rank=16, alpha=32, dropout=0.05
- **Epochs:** 5 (320 steps)
- **Batch Size:** 8
- **Learning Rate:** 0.0002
- **Backend:** Tinker (remote GPU)
- **Expected Duration:** ~17 minutes

### Environment
```
Thinker version : 0.1.0
Python version  : 3.12.3
Platform        : Linux WSL2
Tinker SDK      : 0.3.0
Config file     : thinker/configs/pipeline_scifact.yaml
```

---

## Training Command

### Required Environment Variable
```bash
export TINKER_API_KEY=<your-api-key>
```

### Run Training
```bash
cd /home/home/p/g/North-Shore-AI/tinkerer
python3 -m thinker.cli train --backend tinker
```

### Expected Output
- Training progress with citation validation metrics
- Loss reduction over 320 steps
- Citation invalid rate logging (should remain 0.000 for clean training data)
- Manifest saved to `runs/latest_tinker_adapter.json`
- Provenance log saved to `runs/train_claim-extractor-scifact_<timestamp>.json`

---

## Success Criteria

### Primary Objective
**Eliminate HIGH severity CITATION_INVALID flags**
- Current: 2/50 samples (claims 133, 179)
- Target: 0/50 samples
- Both claims currently fabricate document IDs not in source corpus

### Secondary Objectives
1. **Mean Entailment:** ≥0.50 (current: 0.395, baseline: 0.448)
2. **Overall Pass Rate:** ≥45% (current: 34%, baseline: 38%)
3. **Citation Accuracy:** Maintain ≥95% (current: 96%)
4. **Schema Compliance:** Maintain ≥95% (current: 98-100%)

### Evaluation Metrics to Monitor
- Schema compliance rate
- Citation accuracy rate
- Mean entailment score (DeBERTa-v3 NLI)
- Mean semantic similarity (sentence-transformers)
- Overall semantic pass rate
- Antagonist HIGH severity flags

---

## Post-Training Workflow

### 1. Evaluation
```bash
python3 -m thinker.cli eval
```
Expected output:
- Per-sample metrics with entailment, β₁, chirality
- Aggregate metrics summary
- Results saved to `runs/thinker_eval/scifact_dev_eval.jsonl`

### 2. Antagonist Analysis
```bash
python3 -m thinker.cli antagonist
```
Expected output:
- Structured flags with severity levels
- Issue type distribution
- Results saved to `runs/thinker_eval/scifact_dev_eval_antagonist_flags.jsonl`

### 3. Compare Results
Compare new adapter to:
- **Baseline:** `claim-extractor-scifact-20251118T173307` (schema 100%, citation 96%, entailment 0.448, pass 38%)
- **Previous iteration:** `claim-extractor-scifact-20251118T220454` (schema 98%, citation 96%, entailment 0.395, pass 34%)

Key questions:
- Are HIGH severity CITATION_INVALID flags eliminated?
- Did entailment score improve (target ≥0.50)?
- Did overall pass rate improve (target ≥45%)?

---

## Decision Tree After Training

### If SUCCESS (flags eliminated, entailment ≥0.50, pass ≥45%)
1. Document results in `docs/20251118/training_weight50_results.md`
2. Update README.md, AGENTS.md with new metrics
3. Commit results
4. Proceed to P1 priorities:
   - Antagonist enhancements (embedding anti-neighbor, DeBERTa contradiction)
   - Proposer semantic grounding (contrastive loss)
   - FEVER dataset integration

### If PARTIAL SUCCESS (flags reduced but not eliminated, metrics improved)
1. Document partial improvement
2. Decision: Try weight=10.0 or implement negative example training
3. Analyze which specific patterns were learned vs not learned

### If FAILURE (flags persist, metrics same/worse)
1. Document failure analysis
2. Implement fallback options:
   - **Option A:** Increase to weight=10.0 or weight=20.0
   - **Option B:** Negative example training (augment dataset with invalid citations)
   - **Option C:** Two-stage training (general extraction → citation-focused fine-tuning)
   - **Option D:** Curriculum learning (gradually increase penalty weight over epochs)

---

## Baseline Metrics for Comparison

### Baseline Adapter (claim-extractor-scifact-20251118T173307)
```
Schema Compliance:     100.0% (50/50) ✅
Citation Accuracy:     96.0% (48/50)  ✅
Mean Entailment:       0.448          ⚠️
Entailment Pass Rate:  38.0% (19/50)  ⚠️
Mean Similarity:       0.25           ⚠️
Similarity Pass Rate:  20.0% (10/50)  ⚠️
Overall Pass Rate:     38.0% (19/50)  ⚠️

Antagonist Flags:      46/50 (92%)
HIGH Severity:         2 (4.3%) - CITATION_INVALID
MEDIUM Severity:       44 (95.7%)
β₁:                    0 across all samples
Mean Chirality:        0.561
Mean Fisher-Rao:       16.75
```

### Failed Iteration (claim-extractor-scifact-20251118T220454, weight=2.0)
```
Schema Compliance:     98.0% (49/50)  ⚠️ -2%
Citation Accuracy:     96.0% (48/50)  = unchanged
Mean Entailment:       0.395          ❌ -0.053
Entailment Pass Rate:  34.0% (17/50)  ❌ -4%
Mean Similarity:       0.25           = unchanged
Similarity Pass Rate:  18.0% (9/50)   ❌ -2%
Overall Pass Rate:     34.0% (17/50)  ❌ -4%

Antagonist Flags:      45/50 (90%)
HIGH Severity:         2 (4.3%) - CITATION_INVALID ❌ SAME
MEDIUM Severity:       43 (95.7%)
```

**Critical Finding:** Same 2 HIGH severity flags persist (claims 133, 179)

---

## Known Citation Hallucination Cases

### Claim 133: Podosome Formation
- **Provided Documents:** 38485364, 6969753, 17934082
- **Hallucinated Documents:** 16280642, 12640810
- **Evidence Overlap:** 20%
- **Entailment Score:** 0.0
- **Chirality:** 0.735 (high polarity contradiction)

### Claim 179: Birthweight & Breast Cancer
- **Provided Documents:** 16322674, 27123743, 23557241
- **Hallucinated Documents:** 17450673
- **Evidence Overlap:** 25%
- **Entailment Score:** 0.0
- **Chirality:** 0.675 (high polarity contradiction)

**Pattern:** Both claims fabricate 2-3 document IDs not in the source passage.

---

## Files Modified (Git Status)

### Committed Changes
- `cns-support-models/scripts/train_claim_extractor.py` (e500bb2)
  - Line 41: `CITATION_VALIDITY_WEIGHT` default 2.0 → 5.0

- `AGENTS.md` (34334bd)
  - Added training iteration history
  - Updated Proposer/Antagonist/Synthesizer status
  - Added critical findings and remediation plans

- `README.md` (34334bd)
  - Updated evaluation metrics with training history
  - Expanded Antagonist MVP section
  - Added current status (2025-11-18)

- `ROADMAP.md` (34334bd)
  - Updated snapshot with completed/active/blocked components
  - Added P0/P1/P2 priorities
  - Marked Phase 2 (Tinker integration) as COMPLETE

### No Uncommitted Changes
```
On branch master
nothing to commit, working tree clean
```

---

## References

### Documentation
- **Antagonist MVP Review:** `docs/20251118/antagonist-mvp-review/`
  - COMPREHENSIVE_REVIEW.md
  - FLAG_ANALYSIS.md
  - HIGH_SEVERITY_REVIEW.md
  - TRAINING_RESULTS.md

### Code Files
- **Training Script:** `cns-support-models/scripts/train_claim_extractor.py`
- **Citation Validation:** `thinker/citation_validation.py` (29 tests)
- **Antagonist:** `thinker/antagonist.py` (22 tests)
- **Evaluation:** `thinker/evaluation.py`

### Git Commits
- e500bb2: feat(training): Increase citation validity penalty weight from 2.0 to 5.0
- 34334bd: docs: Update documentation with Antagonist MVP completion and training status

---

## Ready to Proceed

✅ Code configured with weight=5.0
✅ Documentation updated
✅ Git committed (no uncommitted changes)
✅ Environment verified (Thinker 0.1.0, Tinker SDK 0.3.0)
✅ Dataset present (505 examples)
✅ Baseline metrics documented
✅ Success criteria defined
✅ Decision tree prepared

**Next Action:** Set `TINKER_API_KEY` and run training command.

---

**End of Training Ready Document**
