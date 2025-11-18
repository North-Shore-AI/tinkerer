# Citation Validation Training Integration Guide
**Date:** 2025-11-18
**Purpose:** Integrate citation validation into Proposer training to fix citation hallucination

---

## Overview

The `thinker.citation_validation` module provides utilities to detect and penalize citation hallucination during training. This guide explains how to integrate it into the training pipeline.

---

## Module Components

### Core Functions

```python
from thinker.citation_validation import (
    extract_document_ids,      # Extract doc IDs from text
    validate_citations,         # Validate single pair
    batch_validate_citations,   # Validate batch
    compute_citation_penalty,   # Calculate loss penalty
    citation_validation_stats,  # Aggregate statistics
)
```

### Test Coverage

- **29 tests, 100% pass rate** ✅
- Includes real-world validation (claims 133, 179 from HIGH severity review)
- Located in: `thinker/tests/test_citation_validation.py`

---

## Integration Pattern

### For HuggingFace/PEFT Training (LocalPEFTTrainer)

**File:** `thinker/training.py`

**Integration Point:** After tokenization, in the `Trainer` class

**Code Example:**

```python
# In _tokenize_function or custom collator
from thinker.citation_validation import validate_citations, compute_citation_penalty

class CitationAwareDataCollator:
    """Custom collator that validates citations during training."""

    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.citation_weight = config.get("training", {}).get(
            "citation_validity_weight", 2.0
        )
        self.validate_citations = config.get("training", {}).get(
            "validate_citations_during_training", False
        )

    def __call__(self, features):
        # Standard tokenization
        batch = self.tokenizer.pad(features, return_tensors="pt")

        # Add citation validation if enabled
        if self.validate_citations and "prompt" in features[0]:
            prompts = [f["prompt"] for f in features]
            completions = [f["completion"] for f in features]

            from thinker.citation_validation import batch_validate_citations

            results = batch_validate_citations(prompts, completions)

            # Store validation results for custom loss computation
            batch["citation_penalties"] = [
                compute_citation_penalty(r, self.citation_weight)
                for r in results
            ]

        return batch
```

**Custom Trainer with Citation Loss:**

```python
from transformers import Trainer

class CitationAwareTrainer(Trainer):
    """Trainer that adds citation validation penalty to loss."""

    def compute_loss(self, model, inputs, return_outputs=False):
        # Standard loss computation
        outputs = model(**inputs)
        loss = outputs.loss

        # Add citation penalty if available
        if "citation_penalties" in inputs:
            penalties = torch.tensor(
                inputs["citation_penalties"],
                device=loss.device,
                dtype=loss.dtype
            )
            citation_loss = penalties.mean()
            loss = loss + citation_loss

            # Log citation metrics
            self.log({
                "citation_loss": citation_loss.item(),
                "base_loss": (loss - citation_loss).item(),
            })

        return (loss, outputs) if return_outputs else loss
```

**Usage:**

```python
# In LocalPEFTTrainer.train()
from transformers import Trainer

# Replace standard Trainer with CitationAwareTrainer
trainer = CitationAwareTrainer(  # Changed from Trainer
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=CitationAwareDataCollator(tokenizer, config),  # Custom collator
)
```

---

### For Tinker Training (TinkerScriptTrainer)

**File:** `cns-support-models/scripts/train_claim_extractor.py`

**Note:** This script runs on Tinker's remote infrastructure. Integration requires modification of the Tinker training script.

**Recommended Approach:**

1. **Add validation to data preprocessing** (before sending to Tinker)
2. **Filter hallucinating samples** from training data
3. **Log validation statistics** for monitoring

**Code Example:**

```python
# In train_claim_extractor.py, before training
from thinker.citation_validation import batch_validate_citations, citation_validation_stats

def preprocess_training_data(data_path, output_path):
    """Preprocess training data with citation validation."""
    import json
    from pathlib import Path

    with open(data_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    prompts = [s["prompt"] for s in samples]
    completions = [s["completion"] for s in samples]

    # Validate all citations
    results = batch_validate_citations(prompts, completions)
    stats = citation_validation_stats(results)

    print(f"[citation validation] Preprocessing {len(samples)} samples")
    print(f"  Valid rate: {stats['valid_rate']:.2%}")
    print(f"  Mean hallucinations: {stats['mean_hallucinations']:.3f}")
    print(f"  Total hallucinations: {stats['total_hallucinations']}")

    # Option 1: Filter out hallucinating samples (strict)
    clean_samples = [s for s, r in zip(samples, results) if r.is_valid]
    print(f"  Filtered: {len(samples) - len(clean_samples)} samples removed")

    # Option 2: Keep all but add citation_valid flag (for weighted sampling)
    for sample, result in zip(samples, results):
        sample["citation_valid"] = result.is_valid
        sample["hallucination_count"] = result.hallucination_count

    # Write cleaned data
    with open(output_path, "w") as f:
        for sample in clean_samples:  # Or samples if keeping all
            f.write(json.dumps(sample) + "\n")

    return stats

# Call before training
stats = preprocess_training_data(
    "data/processed/scifact_claim_extractor.jsonl",
    "data/processed/scifact_claim_extractor_clean.jsonl"
)
```

---

## Configuration Changes (Already Applied)

**File:** `cns-support-models/configs/claim_extractor_scifact.yaml`

```yaml
training:
  # Citation hallucination fix (2025-11-18)
  cns_claim_evidence_weight: 3.0  # Heavily weight evidence grounding
  citation_validity_weight: 2.0   # Penalize citing non-existent documents
  validate_citations_during_training: true  # Enable validation in training loop
```

**These parameters are ready to use** - just need to be consumed by the training script.

---

## Testing Integration

### Unit Tests

```bash
# Run citation validation tests
$ pytest thinker/tests/test_citation_validation.py -v

# Expected: 29/29 tests PASSED
```

### Integration Test

```python
# Test on real evaluation data
from thinker.citation_validation import batch_validate_citations, citation_validation_stats
import json

# Load evaluation data
with open("runs/thinker_eval/scifact_dev_eval.jsonl") as f:
    data = [json.loads(line) for line in f if line.strip()]

prompts = [d["prompt"] for d in data]
completions = [d["completion"] for d in data]

# Validate
results = batch_validate_citations(prompts, completions)
stats = citation_validation_stats(results)

print(f"Citation validation on {len(results)} samples:")
print(f"  Valid rate: {stats['valid_rate']:.2%}")
print(f"  Mean hallucinations: {stats['mean_hallucinations']:.3f}")

# Expected (before retraining):
#   Valid rate: ~96%  (2/50 invalid = claims 133, 179)
#   Mean hallucinations: 0.06
```

---

## Monitoring & Validation

### During Training

**Log these metrics:**
- `citation_valid_rate`: Fraction of training samples with valid citations
- `mean_hallucinations`: Average hallucinations per sample
- `citation_loss`: Citation penalty contribution to total loss
- `base_loss`: Standard cross-entropy loss

**Expected Trends:**
- `citation_valid_rate`: Should increase from ~96% to >99%
- `mean_hallucinations`: Should decrease from 0.06 to <0.01
- `citation_loss`: Should decrease over epochs as model learns to ground citations

### After Training

**Re-run evaluation with Antagonist:**

```bash
# 1. Train with new config
$ python -m thinker.cli train --backend tinker

# 2. Evaluate
$ python -m thinker.cli eval

# 3. Run antagonist
$ python -m thinker.cli antagonist

# 4. Analyze flags
$ python docs/20251118/antagonist-mvp-review/analyze_flags.py
```

**Success Criteria:**
- `CITATION_INVALID` flags drop from 2/46 (4.3%) to <1/46 (<2%)
- Overall semantic pass increases from 38% to ≥60%
- Mean entailment increases from 0.41 to ≥0.60

---

## Implementation Checklist

### Phase 1: Validation Module (✅ COMPLETE)
- [x] Create `thinker/citation_validation.py`
- [x] Implement `extract_document_ids()`
- [x] Implement `validate_citations()`
- [x] Implement `batch_validate_citations()`
- [x] Implement `compute_citation_penalty()`
- [x] Implement `citation_validation_stats()`
- [x] Create comprehensive tests (29 tests, 100% pass)
- [x] Validate against real data (claims 133, 179)

### Phase 2: Training Integration (IN PROGRESS)
- [x] Document integration patterns for HF/PEFT
- [x] Document integration patterns for Tinker
- [x] Update configuration with validation parameters
- [ ] **TODO:** Modify `thinker/training.py` to add `CitationAwareTrainer`
- [ ] **TODO:** Modify `cns-support-models/scripts/train_claim_extractor.py` for Tinker

### Phase 3: Testing & Validation (NEXT)
- [ ] Integrate citation validation into local training
- [ ] Run smoke test on small dataset
- [ ] Verify citation_loss appears in logs
- [ ] Run full training with Tinker backend
- [ ] Re-evaluate and compare before/after metrics

---

## Next Steps

### Immediate (This Session)
1. Modify `thinker/training.py` to add `CitationAwareTrainer` class
2. Create data preprocessing script for Tinker training
3. Create precision/recall test suite (200 pairs)

### Week 2 (Days 1-3)
1. Test integration with local PEFT training
2. Validate citation loss appears correctly in logs
3. Run Tinker training with validated data
4. Re-evaluate and analyze results

### Week 2 (Days 4-7)
1. Compare before/after metrics
2. Verify citation hallucination <2%
3. Verify semantic pass ≥60%
4. Document results and learnings

---

## Troubleshooting

### Issue: Citation penalties not appearing in loss

**Check:**
1. `validate_citations_during_training: true` in config
2. Custom data collator is being used
3. `citation_penalties` key exists in batch
4. Custom trainer `compute_loss()` is being called

### Issue: Valid rate not improving

**Check:**
1. Training data was preprocessed with validation
2. `citation_validity_weight` is set (default 2.0)
3. Model has enough epochs to learn (≥5)
4. Learning rate is appropriate (2e-4)

### Issue: All samples flagged as invalid

**Check:**
1. Document ID extraction regex is working
2. Prompt format matches expected pattern ("Document 12345:")
3. Completion format matches expected pattern ("CLAIM[c1] (Document 12345):")

---

## References

- **Validation Module:** `thinker/citation_validation.py`
- **Tests:** `thinker/tests/test_citation_validation.py`
- **Config:** `cns-support-models/configs/claim_extractor_scifact.yaml`
- **HIGH Severity Review:** `docs/20251118/antagonist-mvp-review/HIGH_SEVERITY_REVIEW.md`
- **Implementation Summary:** `docs/20251118/antagonist-mvp-review/IMPLEMENTATION_SUMMARY.md`

---

**Status:** Validation module complete, integration patterns documented
**Next:** Implement CitationAwareTrainer and test with local training
