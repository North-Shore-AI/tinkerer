# Thinker ↔ Tinker Roadmap

This roadmap details how the Thinker framework evolves from the current validation-first Hugging Face loop into the orchestration layer for both local and Tinker-backed experiments. It is written for reviewers who need to audit the full workflow—commands, configs, tests, and documentation are all referenced explicitly.

---

## Snapshot · Where We Stand Today

- **Thinker CLI** orchestrates validation (`thinker validate`), training (`thinker train`), evaluation (`thinker eval`), and dataset bootstrap (`thinker data setup`).
- **Datasets**:
  - SciFact helper fully automated (download + convert + validation).
  - FEVER helper now pulls from Zenodo mirrors; conversion script supports JSONL wiki shards.
- **Validation/TDD**: `thinker validate` runs CNS pytest suite + dataset validator (schema or embedding mode).
- **Training/Eval**: HF PEFT backend complete; Tinker backend stub pending. Evaluation script handles SciFact metrics.
- **Docs**: README + `docs/thinker` describe the workflow; `CONTINUATION_PROMPT.md` and this roadmap track next steps.

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

## Phase 2 · Tinker Backend Integration

### 2.1 Implement Tinker Trainer
- ✅ Current shim: Thinker now shells out to `cns-support-models/scripts/train_claim_extractor.py` when `--backend tinker` is specified (pass-through config + log dir).
- Next iteration: replace the shim with a native `TinkerTrainingBackend` that talks to `tinker.ServiceClient` directly.

### 2.2 Evaluation via Tinker Sampling
- Optionally call Tinker sampling client when local checkpoints unavailable.
- Mirror logging: job ID, sample prompts, completions, metrics.

### 2.3 Tests
- Mock Tinker ServiceClient to ensure Thinker builds the correct API calls.
- Add integration test (with fakes) verifying `thinker train --backend tinker` completes on a toy dataset.

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

## Immediate Next Steps
1. **Clean FEVER data**: delete existing raw files, rerun helper, confirm processed dataset exists (update `CONTINUATION_PROMPT.md` once confirmed).
2. **Add FEVER config + tests**: pipeline YAML, fixtures, pytest coverage.
3. **Begin Tinker backend work**: design training client wrapper, plan tests.
4. **Enhance documentation**: once FEVER config is in place, update README + DATA_PIPELINE again.
