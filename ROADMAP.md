# Thinker ↔ Tinker Roadmap

This roadmap details how the Thinker framework evolves from the current validation-first Hugging Face loop into the orchestration layer for both local and Tinker-backed experiments. It is written for reviewers who need to audit the full workflow—commands, configs, tests, and documentation are all referenced explicitly.

---

## Snapshot · Where We Stand Today (Updated 2025-11-18)

### ✅ Completed Components

- **Thinker CLI** - Full orchestration for validation, training, evaluation, antagonist analysis, and data setup
- **Tinker Backend** - ✅ COMPLETE: Production-ready integration with citation validation, manifest generation, telemetry
- **Antagonist MVP** - ✅ COMPLETE: 92% flagging rate, 4 issue types, 22 tests, comprehensive documentation
- **4-Stage Semantic Validation** - ✅ OPERATIONAL: Citation → Entailment → Similarity → Paraphrase
- **Topology Instrumentation** - ✅ WORKING: β₁ (Betti numbers), chirality, Fisher-Rao distance
- **Dashboard & Telemetry** - ✅ COMPLETE: Multi-run visualization, training/eval/antagonist charts
- **Datasets**:
  - SciFact: Fully automated (download + convert + validation)
  - FEVER: Helper pulls from Zenodo mirrors, conversion script supports JSONL wiki shards

### ⚠️ Active Critical Work (P0)

- **Citation Hallucination Fix** - Training with `citation_validity_weight=5.0` to eliminate HIGH severity CITATION_INVALID cases
  - Status: Code committed (commit `e500bb2`), training run pending
  - Previous attempt (weight=2.0) FAILED to eliminate hallucinations
  - Success criteria: Eliminate 2 HIGH severity flags, mean entailment ≥0.50, overall pass ≥45%

### 🔴 Blocked Components

- **Synthesizer Agent** - Blocked until Proposer reaches ≥60% semantic quality (currently 34-38%)
  - Blocking issues: Citation hallucinations, weak entailment (0.395-0.448)
  - Unblocking criteria: Mean entailment ≥0.60, HIGH severity flags eliminated

### 📊 Current Metrics (Baseline: claim-extractor-scifact-20251118T173307)

```
Schema Compliance:     100% ✅
Citation Accuracy:     96% ✅
Mean Entailment:       0.448 ⚠️ (target ≥0.75)
Overall Semantic Pass: 38% ⚠️ (target ≥60%)
Antagonist Flags:      46/50 (92%), 2 HIGH severity
β₁ (cycles):           0 across all samples
Mean Chirality:        0.561
```

---

## Phase 1 · Finish Data + Validation Coverage

### 1.1 FEVER Reliability
1. Delete corrupted raw files and rerun `python -m thinker.cli data setup --dataset fever --skip-validation` to confirm a clean processed JSONL.
2. Add FEVER fixtures/tests:
   - Sample FEVER claims + wiki lines under `cns-support-models/tests/fixtures/fever_*`.
   - Tests similar to SciFact (CLAIM parsing, converter CLI).
3. Create `thinker/configs/pipeline_fever.yaml` pointing at FEVER paths (data validation, training, evaluation).
4. Update docs (README, DATA_PIPELINE) to mention FEVER config and commands.

### 1.2 Enhanced Validation Options
- Extend `DatasetValidationConfig` with:
  - Regex checks, numeric bounds, Hypothesis-driven validators.
  - Per-dataset defaults (SciFact vs FEVER).
- Add CLI options for dataset validator script to select dataset-specific schemas.

---

## Phase 2 · Tinker Backend Integration ✅ COMPLETE

### 2.1 Implement Tinker Trainer ✅ DONE
- ✅ Tinker backend functional via shim to `cns-support-models/scripts/train_claim_extractor.py`
- ✅ Citation validation integrated with configurable penalty weights
- ✅ Manifest generation (`runs/latest_tinker_adapter.json`)
- ✅ Provenance logging to `runs/train_*.json`
- ✅ Telemetry: loss, citation_invalid_rate, timestamps at each step
- Future enhancement (P2): Native `TinkerTrainingBackend` using `tinker.ServiceClient` directly

### 2.2 Evaluation via Tinker Sampling ✅ DONE
- ✅ Tinker sampling client integrated in `thinker/evaluation.py`
- ✅ Loads tokenizer via API, samples from adapter in manifest
- ✅ Logs job ID, sample prompts, completions, metrics
- ✅ Per-sample topology/chirality instrumentation
- ✅ Live progress logging (`sample N/50 | entailment | β₁ | chirality`)

### 2.3 Tests ✅ PARTIALLY COMPLETE
- ✅ 22 tests for Antagonist
- ✅ Citation validation: 29 tests
- ✅ Integration tested via real training runs
- ⏳ Future: Mock Tinker ServiceClient for unit tests (P2)

---

## Phase 3 · Experiment Playbooks

### 3.1 SciFact Baseline (HF)
1. `python -m thinker.cli data setup --dataset scifact --validation-mode embedding --similarity-threshold 0.7`
2. `python -m thinker.cli validate`
3. `python -m thinker.cli train --backend hf_peft`
4. `python -m thinker.cli eval`
5. Log metrics + configs in `cns-support-models/notes/claim_extractor.md`

### 3.2 FEVER Baseline (HF)
- Same flow but using FEVER config; evaluate impacts of larger dataset and NEI cases.

### 3.3 Backend Comparison
- Run identical config on HF and Tinker backends; compare metrics, runtime, resource usage.
- Document differences in notes + run metadata.

---

## Phase 4 · Critic Integration & Advanced Validation

### 4.1 Extend Dataset Validator
- Add plugin system for custom validators (e.g., relation semantics, critic-specific checks).
- Support Hypothesis property tests triggered via Thinker config.

### 4.2 Critic Pipeline Hooks
- Integrate logic/grounding critics once they exist, ensuring Thinker enforces the same validation gate before critic training.
- Provide CLI flags to run critic training/evaluation.

---

## Documentation & Audit Trail

- **README** – now references Thinker CLI; keep updated when new commands/configs land.
- **docs/thinker/THINKER_SPEC.md** – update whenever CLI/validation contracts change.
- **docs/thinker/DATA_PIPELINE.md** – the canonical guide to data setup (SciFact/FEVER, caching, troubleshooting).
- **RUN METADATA** – ensure each Thinker run writes JSON metadata (config paths, dataset hashes, metrics) in `runs/thinker/`.
- **CONTINUATION_PROMPT.md** – short task list for next engineer; update after each major milestone.

---

## Immediate Next Steps (2025-11-18)

### P0 - Critical (This Week)
1. **✅ DONE: Documentation updates** - README.md, AGENTS.md, ROADMAP.md updated with latest status
2. **🔬 IN PROGRESS: Training with weight=5.0** - Eliminate citation hallucinations
   - Run: `python -m thinker.cli train --backend tinker`
   - Expected duration: ~17 minutes (320 steps, 5 epochs)
   - Success criteria: 2 HIGH severity flags → 0, mean entailment ≥0.50
3. **⏳ NEXT: Evaluate training results** - Run full evaluation + antagonist analysis
   - `python -m thinker.cli eval`
   - `python -m thinker.cli antagonist`
   - Compare metrics to baseline and weight=2.0 iteration
4. **⏳ DECISION POINT: Weight=5.0 outcome**
   - If SUCCESS: Document results, proceed to P1 priorities
   - If FAILURE: Escalate to weight=10.0 or implement negative example training

### P1 - High Priority (Next 1-2 Weeks)
1. **Antagonist enhancements**:
   - Embedding anti-neighbor retrieval for counter-evidence
   - DeBERTa contradiction scoring for POLARITY_CONTRADICTION
   - 200-pair synthetic contradiction test suite (precision/recall)
2. **Proposer semantic grounding**:
   - Contrastive loss integration (if weight=5.0 insufficient)
   - Scale to 1000+ training examples
   - Consider LoRA rank increase (16 → 32)
3. **FEVER dataset**: Add fixtures, tests, pipeline config

### P2 - Medium Priority (Next 2-4 Weeks)
1. **Tinker backend native implementation**: Replace shim with `TinkerTrainingBackend` using `ServiceClient` directly
2. **Enhanced validation options**: Regex checks, numeric bounds, per-dataset defaults
3. **Synthesizer prep** (once Proposer unblocks): Critic interfaces, SNO manifest schema
