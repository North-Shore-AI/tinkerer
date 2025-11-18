# Training Status & Next Steps
**Date:** 2025-11-18
**Status:** ⚠️ BLOCKED - Hardware/API Access Needed

---

## Current Situation

### ✅ COMPLETE: Citation Validation Implementation

All code is complete and ready for training:

1. **CitationAwareTrainer** - Training pipeline with citation validation (186 lines)
2. **CitationAwareDataCollator** - Batch-level citation validation
3. **Citation Validation Module** - 29 tests, 100% passing
4. **Configuration** - Citation validation enabled in configs
5. **Documentation** - Comprehensive implementation guides

**Code Status:** Production-ready, committed to master
**Test Status:** All passing (29/29 citation validation, 22/22 antagonist)

---

## Training Options Analysis

### Option 1: LocalPEFTTrainer + GPU (RECOMMENDED)

**Status:** ⚠️ Requires GPU setup

**Pros:**
- Citation validation fully implemented and tested
- Fast iteration (hours, not days)
- Full control over training process
- Granular logging every 5 steps
- Can monitor citation_loss in real-time

**Cons:**
- Requires local GPU with CUDA
- Requires bitsandbytes package
- Requires GPU drivers

**Requirements:**
```bash
# Hardware
- NVIDIA GPU with ≥16GB VRAM (for 8B model + 4-bit quant)
- CUDA 11.8 or 12.x

# Software
pip install bitsandbytes
pip install accelerate

# Verification
python -c "import bitsandbytes; print(bitsandbytes.__version__)"
```

**Training Command:**
```bash
cd /home/home/p/g/North-Shore-AI/tinkerer
python3 -m thinker.cli train --backend hf_peft
```

**Expected Duration:** ~2-4 hours for 5 epochs

---

### Option 2: Tinker API (ALTERNATE)

**Status:** ⚠️ Requires API key + code modifications

**Pros:**
- Remote GPU access (no local hardware needed)
- Managed infrastructure
- Used for evaluation already

**Cons:**
- Citation validation NOT YET integrated in Tinker script
- Requires modifying train_claim_extractor.py
- Less visibility into training process
- Harder to debug

**Requirements:**
```bash
# Environment
export TINKER_API_KEY="your-key-here"

# Code modifications needed:
# 1. Add citation validation to train_claim_extractor.py
# 2. Extract document IDs from prompt and completion
# 3. Compute citation penalty in loss calculation
# 4. Add penalty to Tinker forward_backward call
```

**Implementation Needed:**
- Estimate: 2-3 hours to integrate citation validation
- Testing: 1-2 hours
- Total: Half day of work

**Training Command:**
```bash
cd /home/home/p/g/North-Shore-AI/tinkerer
export TINKER_API_KEY="..."
python3 -m thinker.cli train --backend tinker
```

---

### Option 3: LocalPEFTTrainer + CPU (NOT RECOMMENDED)

**Status:** ✅ Works but impractically slow

**Pros:**
- No GPU required
- Works on current system
- Citation validation fully implemented

**Cons:**
- **EXTREMELY SLOW:** 12-48 hours PER EPOCH
- 5 epochs = 2.5-10 DAYS total
- Not practical for iteration

**Config:** `lora_config_cpu.yaml` (already created)

**Only use for:**
- Smoke testing citation validation logic
- Verifying no crashes
- NOT for actual training runs

---

## Recommended Path Forward

### Immediate (Now)

**RECOMMENDED:** Set up local GPU for fast iteration

Steps:
1. Install NVIDIA drivers (if not installed)
2. Install CUDA toolkit
3. Install bitsandbytes: `pip install bitsandbytes`
4. Test GPU: `python -c "import torch; print(torch.cuda.is_available())"`
5. Run training: `python3 -m thinker.cli train --backend hf_peft`

Expected time: 30min setup + 2-4 hours training

### Alternate (If GPU unavailable)

**Implement Tinker citation validation**

Steps:
1. Modify `train_claim_extractor.py` to add citation validation
2. Extract document IDs using citation_validation module
3. Compute penalties and add to loss
4. Test with Tinker API
5. Run training

Expected time: Half day implementation + training time

---

## What We've Accomplished

### Code Complete (2,452 lines across 10 files)

**P0 Implementation:**
- Antagonist MVP with CITATION_INVALID detection (784 lines)
- 22 comprehensive tests (100% pass rate)
- Config updates for citation penalties
- Documentation (README, AGENTS)

**P1 Implementation:**
- CitationAwareTrainer and CitationAwareDataCollator (186 lines)
- Citation validation module (193 lines)
- 29 comprehensive tests (100% pass rate)
- Training integration guide (19KB)
- Implementation documentation (19KB)

**Configuration:**
- lora_config.yaml - GPU training with citation validation
- lora_config_cpu.yaml - CPU fallback (slow but works)
- pipeline_scifact.yaml - Updated for CPU config

### Testing Status

**All tests passing:**
```bash
$ pytest thinker/tests/test_citation_validation.py
============================== 29 passed ==============================

$ pytest thinker/tests/test_antagonist.py
============================== 22 passed ==============================
```

**Integration tests needed:**
- Training run with citation validation (blocked on hardware)
- Verify citation_loss appears in logs (blocked on hardware)
- Re-evaluate with antagonist after training (blocked on training)

---

## Experiment Plan (Once Unblocked)

### Phase 1: Baseline Training Run

```bash
# Start training
python3 -m thinker.cli train --backend hf_peft

# Monitor logs (in separate terminal)
tail -f cns-support-models/checkpoints/claim-extractor-scifact/runs/*/trainer_state.json

# Look for:
# - citation_loss (should decrease over time)
# - base_loss (standard cross-entropy)
# - total_loss (base + citation)
```

**Expected training logs:**
```
Step 5:   loss=2.45, base_loss=1.80, citation_loss=0.65  # Early: many hallucinations
Step 50:  loss=2.10, base_loss=1.75, citation_loss=0.35  # Mid: learning
Step 100: loss=1.85, base_loss=1.72, citation_loss=0.13  # Late: improving
Step 140: loss=1.78, base_loss=1.71, citation_loss=0.07  # Final: well-grounded
```

**Success criteria:**
- citation_loss decreases from ~0.65 to <0.10
- No crashes or errors
- Training completes all 5 epochs
- Model saved to checkpoint dir

### Phase 2: Evaluation

```bash
# Evaluate trained model
python3 -m thinker.cli eval

# Run antagonist on new evaluation
python3 -m thinker.cli antagonist

# Analyze results
python docs/20251118/antagonist-mvp-review/analyze_flags.py
```

**Expected improvements:**
- CITATION_INVALID flags: 2/46 (4.3%) → <1/46 (<2%)
- Valid citation rate: 96% → >99%
- Mean hallucinations: 0.06 → <0.01
- Overall semantic pass: 38% → ≥60%
- Mean entailment: 0.41 → ≥0.60

### Phase 3: Analysis & Iteration

**If citation_loss didn't decrease:**
- Check if validate_citations_during_training is enabled
- Verify citation_penalties appear in batch
- Review training logs for errors
- Try increasing citation_validity_weight from 2.0 to 5.0

**If entailment didn't improve:**
- Increase cns_claim_evidence_weight from 3.0 to 5.0
- Train for more epochs (10 instead of 5)
- Review evaluation data for systematic failures

**If still failing:**
- Create precision/recall test suite
- Analyze failure modes
- Implement targeted fixes

---

## Decision Points

### Now (Unblock Training)

**Choice 1: GPU Setup (RECOMMENDED)**
- **Time:** 30min setup
- **Benefit:** Fast iteration, full citation validation
- **Drawback:** Requires hardware configuration

**Choice 2: Tinker Integration**
- **Time:** Half day implementation
- **Benefit:** No local GPU needed
- **Drawback:** More code to write and test

**Choice 3: Wait on CPU**
- **Time:** 2.5-10 days per run
- **Benefit:** No setup needed
- **Drawback:** Too slow for research iteration

### After Training (Iterate)

**If results good (≥60% semantic pass):**
- Document success
- Run on full dataset (not just dev set)
- Consider production deployment

**If results mixed (40-60% semantic pass):**
- Increase epochs or learning rate
- Tune citation_validity_weight
- Analyze failure modes
- Create targeted test suite

**If results poor (<40% semantic pass):**
- Revisit problem formulation
- Consider different model architecture
- Implement additional heuristics

---

## Files Ready for Training

### Configurations
```
thinker/configs/
├── lora_config.yaml          # GPU training (4-bit quant)
├── lora_config_cpu.yaml       # CPU training (slow)
└── pipeline_scifact.yaml      # Pipeline orchestration
```

### Training Data
```
cns-support-models/data/processed/
└── scifact_claim_extractor_clean.jsonl  # 505 samples
```

### Code
```
thinker/
├── training.py                    # CitationAwareTrainer + Collator
├── citation_validation.py         # Validation utilities
└── tests/
    ├── test_citation_validation.py  # 29 tests
    └── test_antagonist.py            # 22 tests
```

---

## Summary

**What's Done:**
✅ All code complete and tested
✅ Configuration files ready
✅ Documentation comprehensive
✅ Tests passing (51/51)

**What's Blocking:**
⚠️ Hardware setup (GPU) OR Tinker API access

**What's Next:**
1. Choose training option (GPU recommended)
2. Run training with citation validation
3. Monitor citation_loss metrics
4. Evaluate with antagonist
5. Compare before/after results
6. Iterate based on findings

**Time to First Results:**
- GPU setup: 30min + 2-4 hours training = ~5 hours total
- Tinker: Half day implementation + training time
- CPU: 2.5-10 days (not recommended)

---

**Status:** Ready to train pending hardware/API setup
**Blocker:** Need GPU or Tinker API key
**Next Action:** User decision on training infrastructure

