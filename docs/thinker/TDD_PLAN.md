# Thinker TDD & Test Strategy

**Revision:** 2025-11-10  
**Owners:** CNS Support Models Team  
**Related Docs:** `docs/thinker/THINKER_SPEC.md`, ADRs 0001-0003, `brainstorm/20251110/003_tinker_tdd_claude.md`

---

## 1. Objectives

1. **Catch data/representation defects before training.** All schema, golden, and property tests must run in `<60 s` so developers never skip them.  
2. **Guarantee pipeline regressions surface immediately.** CLI + backend interfaces have comprehensive unit and integration coverage.  
3. **Provide confidence for CI/CD.** Every PR runs the same validation as `thinker validate`; nightly jobs perform extended evaluations.  
4. **Enable reproducibility.** Golden fixtures and metadata snapshots allow deterministic comparisons across runs/backends.

---

## 2. Test Taxonomy

| Level | Purpose | Example Areas | Tooling |
| ----- | ------- | ------------- | ------- |
| **Unit** | Verify individual modules/functions | Config loader, schema validator, CLI parser, tokenizer logic | `pytest`, `unittest.mock` |
| **Property** | Validate invariants across randomized data | JSONL schema, CLAIM parser, prompt/label masking | `hypothesis` |
| **Integration** | Ensure multi-module flows operate correctly | Validation stage + dataset validator, HF training with synthetic data, evaluation pipeline with stubbed model | `pytest`, temporary directories, fakes |
| **End-to-End (E2E)** | Exercise full CLI commands | `thinker run` on small fixture dataset (HF backend) | `pytest + subprocess` or direct CLI invocation |
| **Golden / Regression** | Catch drift in converters/outputs | Sample SciFact conversion outputs, evaluation JSONL snapshots | fixture files under `tests/golden/` |
| **Performance/Monitoring** | Track runtime budgets | Validation runtime, training step counts (future) | Telemetry hooks + assertions |

---

## 3. Coverage Map

### 3.1 Module → Tests Mapping

| Module | Unit Tests | Property Tests | Integration Tests |
| ------ | ---------- | -------------- | ----------------- |
| `thinker.config` | `test_config_loads_required_sections`, `test_relative_paths_resolved`, `test_schema_field_validation` | `test_schema_field_random_payloads` | Config + CLI e2e ensures config drives pipeline |
| `thinker.validation.TestSuiteRunner` | `test_runner_pass_through_args`, `test_runner_handles_disabled` | — | `test_validate_stage_runs_pytest_and_dataset` |
| `thinker.validation.DatasetValidator` | `test_valid_payload_passes`, `test_missing_field_reports_error`, `test_optional_fields` | `test_random_payloads_respect_schema` | Combined with CLI `validate` command |
| `thinker.training.LocalPEFTTrainer` | `test_load_yaml_paths`, `test_apply_lora_sets_trainable_params`, `test_training_args_without_eval` | `test_tokenizer_mask_prompt_property` | `test_hf_training_on_tiny_dataset_overfits` |
| `thinker.training.create_training_backend` | `test_backend_selection`, `test_unknown_backend_raises` | — | E2E run w/ backend override |
| `thinker.evaluation.Evaluator` | `test_prompt_builder_includes_corpus`, `test_metrics_empty_lists` | `test_semantic_match_commutativity` | `test_eval_with_stub_model_returns_metrics` |
| `thinker.pipeline.ThinkerPipeline` | `test_train_enforces_validation_by_default`, `test_run_executes_all_stages` | — | CLI E2E |
| `thinker.cli` | `test_parser_requires_command`, `test_train_backend_argument`, `test_eval_skip_validation_flag` | — | Subprocess CLI invocation |
| `thinker.claim_schema` | `test_parse_claims`, `test_parse_relations`, `test_enforce_c1`, `test_roundtrip` | `test_random_claim_roundtrip` | Indirect via evaluation stage |

### 3.2 External Scripts (CNS)
- `scripts/convert_scifact.py`: Already requires integration tests ensuring deterministic completions (CLAIM[c1] exact, evidence verbatim).  
- `scripts/train_peft.py`, `scripts/eval_scifact.py`: Covered indirectly via Thinker; maintain legacy unit tests for utility functions if reused.

---

## 4. Test Suites Detail

### 4.1 Unit Tests (Pytest)
- **Config Loader:** Mock YAML files using `tmp_path`. Assert that relative paths resolve correctly and missing keys raise `KeyError`.  
- **SchemaField Validation:** Parameterized tests for `string`, `array`, `object`, optional fields, `allow_empty`.  
- **CLI Parser:** Use `thinker.cli.build_parser` with `parse_args` to ensure defaults (e.g., default config path) and error messaging.  
- **Training Tokenizer:** For deterministic samples, check that prompt tokens are masked to `-100` and completions remain intact.  
- **Evaluation Metrics:** Provide small prompts/completions to test `c1_exact_match` and `evidence_semantic_match` calculations.

### 4.2 Property Tests (Hypothesis)
- **Dataset Schema:** Randomly generate payloads with valid/invalid fields to ensure validator catches or accepts appropriately.  
- **CLAIM Parser:** Random strings with `CLAIM[...]` patterns to confirm regex handles whitespace/case.  
- **Prompt Masking:** Hypothesis test ensuring for any prompt/completion, masked label positions never leak prompt tokens.

### 4.3 Integration Tests
1. **`thinker.validation` Integration:** Use a temporary pytest project with one passing test; run `ThinkerPipeline.validate()` to ensure both pytest and dataset validation run (use temporary JSONL fixture).  
2. **HF Training Overfit Test:** Create synthetic JSONL with 2 samples (tiny vocabulary). Ensure `LocalPEFTTrainer.train()` runs a single epoch and produces checkpoint directory with model/tokenizer.  
3. **Evaluation with Stub Model:** Mock `AutoModelForCausalLM.generate` to return deterministic tokens; ensure evaluator computes metrics and writes JSONL output.  
4. **Pipeline End-to-End (Local Backend):** With sample data + minimal config, run `ThinkerPipeline.run(backend="hf_peft")` with `gradient_accumulation_steps=1`, `num_epochs=1`, `max_samples=1` to assert metadata output exists.

### 4.4 CLI / E2E Tests
- Use `pytest` to call `thinker.cli.main` with arguments (`["validate"]`, `["train", "--skip-validation"]`, etc.) capturing exit codes.  
- Use `subprocess.run(["python", "-m", "thinker.cli", ...])` in one test that exercises actual CLI packaging (ensures __main__ works).  
- Fixture environment sets `PYTHONPATH` appropriately to import `thinker` package.

### 4.5 Golden Fixtures
- **Input:** `thinker/tests/fixtures/sample_data.jsonl` (already present). Add more fixtures:
  - `tests/golden/scifact_sample_claim.json` – canonical prompt/completion pair.  
  - `tests/golden/pipeline_config.yaml` – reference config used in tests.  
- **Output:** For evaluation tests, capture expected JSONL result to compare with actual (allowing for deterministic stub).

### 4.6 Dataset Regression Tests
- `tests/test_data_quality.py` (adapted from TDD docs):
  - `test_c1_matches_gold`: assert every completion’s `CLAIM[c1]` equals hypothesis string from metadata.  
  - `test_evidence_sentences_exist_in_corpus`: cross-check completions against corpus sentences.  
  - `test_relations_reference_existing_claims`.  
  - Use Hypothesis to sample random entries ensuring invariants hold across dataset.

---

## 5. Test Data Management

- **Location:** All fixtures under `thinker/tests/fixtures` or `cns-support-models/tests/golden`.  
- **Size Limits:** Keep fixture JSONL files ≤ 50 KB so they can be committed.  
- **Regeneration:** Provide scripts for regenerating golden data when schema changes (document steps).  
- **Versioning:** Run metadata should record fixture versions/SHAs.

---

## 6. CI / Automation Plan

| Pipeline | Trigger | Steps |
| -------- | ------- | ----- |
| **Pull Request CI** | Every PR | 1) `python -m thinker.cli validate` (pytest + dataset checks). 2) `python -m thinker.cli train --backend=hf_peft --skip-validation --config thinker/tests/fixtures/pipeline_ci.yaml` (mini run). 3) `python -m thinker.cli eval --skip-validation --config thinker/tests/fixtures/pipeline_ci.yaml`. |
| **Nightly** | 1× per day | 1) Full dataset validation on latest SciFact. 2) `thinker run --backend=hf_peft` (longer run). 3) `thinker run --backend=tinker` (when backend implemented) flagged as optional. 4) Upload run metadata artifacts. |
| **Release** | Manual | 1) Full suite including property tests, golden diff check, evaluation with >50 samples. 2) Validate documentation links. |

**CI Considerations**
- Use caching (virtualenv + HF cache) to reduce runtime.  
- For GPU-required tests, mark as optional or skip unless GPU available (CI env variable).  
- Provide `ci` extras in requirements (`pip install -e .[ci]`).

---

## 7. Tooling & Libraries

- **Pytest Plugins:** `pytest-cov`, `pytest-timeout` (fail fast if test hangs).  
- **Hypothesis:** property-based tests for schema + parser.  
- **Coverage:** Target ≥85% for Thinker modules (run `pytest --cov=thinker`).  
- **Linters:** `ruff` or `flake8` to maintain code quality (documented in dev workflow).

---

## 8. Test Data Sources & References

- SciFact JSONL files (`cns-support-models/data/raw/scifact`).  
- Corpus data for evidence cross-checks.  
- Additional corpora (FEVER) once pipeline generalizes; create separate fixtures.  
- Brainstorm docs for acceptance criteria (e.g., evidence recall improvement, relation accuracy).

---

## 9. Backlog / Enhancements

1. **Loss-weighted trainer tests:** Add tests ensuring token-level weights applied correctly once custom Trainer implemented.  
2. **Tinker backend fakes:** Build mock Tinker client to simulate `forward_backward`/`optim_step` for integration tests.  
3. **Evaluation semantics:** Add embedding-based semantic match test once module exists.  
4. **Performance regression suite:** Track validation + training runtime budgets to detect slowdowns.  
5. **Docs validation:** Add tests ensuring ADR/spec references remain valid (link checker).  
6. **Mutation testing:** Evaluate `mutmut` or similar to ensure tests catch injected faults.

---

## 10. Adoption Checklist

- [ ] Migrate existing CNS pytest suites into repository, updated to new layout.  
- [ ] Add `thinker/tests` modules covering pipeline components (initial 4 tests already present).  
- [ ] Create CI fixture config for quick `thinker run`.  
- [ ] Enforce `thinker validate` pre-commit hook (optional).  
- [ ] Track coverage in CI reports; block merges if coverage drops below threshold.

---

## 11. Appendix: Sample Test Matrix

| Test ID | Description | Type | Files |
| ------- | ----------- | ---- | ----- |
| T-001 | Config loader resolves relative paths | Unit | `thinker/tests/test_config.py` |
| T-005 | Dataset validator rejects missing completion | Unit | `thinker/tests/test_validation.py` |
| T-010 | Pipeline enforces validation before training | Integration | `thinker/tests/test_pipeline.py` |
| T-020 | HF backend saves checkpoint | Integration | `tests/integration/test_hf_training.py` (todo) |
| T-030 | CLI run command orchestrates stages | E2E | `tests/e2e/test_cli_run.py` (todo) |
| T-040 | Convert SciFact ensures CLAIM[c1] literal | Golden | `cns-support-models/tests/test_scifact_conversion.py` |
| T-050 | Hypothesis property for CLAIM parser | Property | `thinker/tests/test_claim_schema_property.py` |
| T-060 | Dataset validation max_examples limit | Unit | `thinker/tests/test_validation.py` |
| T-070 | Evaluation metrics accumulate correctly | Unit | `thinker/tests/test_evaluation_metrics.py` |

---

By following this plan, Thinker becomes a rigorously tested orchestrator that fulfills the “test before Tinker” mandate and provides clear confidence signals at every layer. Continuous expansion of fixtures and property tests will ensure data integrity scales alongside model complexity.
