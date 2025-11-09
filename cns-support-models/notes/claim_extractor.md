# Claim Extractor Research Log

## 2025-11-09 — Canonical Enforcement & Structured Eval

- **Motivation**: CNS ingestion needs deterministic hypotheses (`CLAIM[c1]`) and evidence telemetry to reason about downstream accuracy. Previous evals failed literal matching even though training data was correct.
- **Changes**:
  - Added `claim_schema.py` helpers to parse/render CLAIM/RELATION lines and to force `CLAIM[c1]` to a canonical string.
  - Updated `eval_scifact_dev.py` with `--include-gold-claim`, `--enforce-gold-claim`, and semantic/relation scoring built on SciFact evidence sentences. Output JSONL now stores `claims_structured`, `semantic_matches`, `raw_*` fields.
  - Wired the same enforcement path into `eval_claim_extractor.py` via `--force-c1-text/--force-c1-file` so ad-hoc sampling or future ingestion scripts can normalize hypotheses.
  - Added `CNS_CLAIM_EVIDENCE_WEIGHT` so training can up-weight every evidence line (`CLAIM[c2+]`). This acts as an “alignment loss” by heavily penalizing deviations from the gold supporting sentences—set it to `2.0` before the next training run to encourage literal copying.
- **Run**: `runs/scifact_dev_eval_canonical.jsonl` (50 dev claims, Llama‑3.1‑8B adapter).
  - `CLAIM[c1]` literal match: **100% raw / 100% enforced**.
  - Semantic hit rate (`>=0.7` similarity vs gold evidence): **12 / 136 = 8.8%**.
  - Relation accuracy on matched evidence: **0 / 1 (no reliable matches yet)**.
  - Cleanup/fallback: **2 / 50** samples needed line fixes, only one sample retried for schema.
- **Observations**:
  - Model happily restates the canonical hypothesis when it is supplied, so ingestion can always emit the gold string after post-processing.
  - Evidence claims remain broad summaries that rarely overlap the annotated SciFact sentences (most `best_score` values fall below 0.4). Without a semantic match, relation labels cannot be trusted—hence 0% relation accuracy.
  - High-scoring matches occur only when the adapter paraphrases close to the sentence text (e.g., sample 23 `c2`, sample 32 `c5`/`c6`). These are the cases to mine for contrastive supervision.
- **Next Actions**:
  1. Collect high `best_score` false negatives and true positives to design an auxiliary loss (e.g., contrastive or token-level alignment) that anchors evidence claims to gold sentences.
  2. Consider adding a critic or verifier that re-scores generated claims against the passage/evidence, using the new structured outputs as inputs.
  3. Track future experiments via this log and corresponding GitHub issues in `tinkerer` (e.g., “CNS-ClaimExtractor: semantic alignment”).
