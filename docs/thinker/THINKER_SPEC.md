# Thinker Specification · Transformers + PEFT + TDD Orchestrator

**Revision:** 2025-11-10  
**Owners:** CNS Support Models Team  
**Sources:**  
- `brainstorm/20251110/transformers-peft-tdd/01_IMPLEMENTATION_GUIDE.md` (implementation steps & configs)  
- `brainstorm/20251110/003_tinker_tdd_claude.md` (TDD mandate)  
- `brainstorm/20251110/0010_claudeCode_feedback.md`, `0012_tinker_vs_diy.md`, `0013_tinker_diy_thoughts.md` (platform analysis)  
- ADRs `0001-0003` (orchestration, TDD gate, dual backends)

---

## 1. Purpose & Scope

Thinker is the local orchestration layer that enforces **test-first data validation**, **backend-agnostic training**, and **deterministic evaluation** for CNS support-model experiments. It wraps existing scripts/configs into a declarative pipeline so we can:

1. Guarantee dataset quality before consuming GPU cycles (local or Tinker).  
2. Run the exact same training configuration against **Hugging Face + PEFT** (local QLoRA) or **Tinker LoRA service** without changing code.  
3. Capture run metadata, dataset hashes, and evaluation outputs for reproducibility.  
4. Provide a CLI contract for contributors and CI (validate → train → eval).

This specification defines pipeline stages, component responsibilities, interfaces, and operational requirements so that documentation, code, and tests remain synchronized.

---

## 2. Goals & Non-Goals

### 2.1 Goals
- **Validation-first workflow.** No training or eval step runs unless schema + pytest suites succeed.  
- **Backend abstraction.** Training is described once (YAML) and executed via selected backend (HF PEFT, Tinker).  
- **Deterministic configs.** All file paths, seeds, and hyperparameters resolved relative to pipeline config.  
- **Full observability.** Runs emit JSON metadata: git commit, dataset SHA256, config fingerprint, backend, exit status.  
- **Extensibility.** Additional stages (critics, dataset converters) plug into Thinker via typed interfaces.

### 2.2 Non-Goals
- Thinker does **not** replace tinker ServiceClient APIs; it orchestrates calls and ensures preconditions.  
- Thinker does **not** manage hardware provisioning; it assumes HF/Tinker backends have credentials/quotas available.  
- Thinker does **not** implement UI dashboards (future Elixir “Thinker” may do so).  
- Thinker does **not** enforce experiment tracking integrations (W&B, MLflow) but exposes hooks.

---

## 3. Usage Targets

| Persona | Needs | Thinker Support |
| ------- | ----- | --------------- |
| ML Engineer | Fast loop on SciFact data without burning GPU minutes | local HF backend + validation gate |
| Research Scientist | Swap between local smoke checks and Tinker-scale runs | backend override, shared configs |
| CI/CD | Enforce tests and produce regression artifacts | `thinker validate` & `thinker run --backend=hf_peft` in pipelines |
| Reviewer | Inspect reproducibility info | Run metadata JSON + log directory |

---

## 4. Pipeline Overview

```
┌──────────────┐   ┌─────────────┐   ┌─────────────────┐   ┌───────────────┐
│ Config Loader│─▶│ Validation   │─▶│ Training Backend │─▶│ Evaluation     │
└──────────────┘   │ (pytest +    │   │ (HF PEFT /      │   │ (SciFact eval │
                   │ data schema) │   │  Tinker LoRA)   │   │  + metrics)   │
                   └─────────────┘   └─────────────────┘   └───────────────┘
                           │                 │                     │
                           │                 └───────┬─────────────┘
                           │                         ▼
                           └──────────▶ Run Metadata / Logs
```

### 4.1 Stage Contracts
1. **Config Loader**
   - Input: `pipeline.yaml` (explicit paths relative to file).  
   - Output: Normalized `PipelineConfig` dataclass instances (tests, validation, training, evaluation).  
   - Failure: Missing files, unresolved relative paths, unsupported backend names.
2. **Validation Stage**
   - Runs pytest with configured markers/args.  
   - Runs dataset validator on JSONL file with schema definitions (required fields, types, emptiness, etc.).  
   - Failure aborts pipeline; training/eval cannot run unless `--skip-validation` is passed (debug only).  
3. **Training Stage**
   - Accepts backend enum.  
   - HF path: loads YAML (LoRA config), resolves dataset/checkpoint paths, runs Transformers `Trainer` with QLoRA support.  
   - Tinker path (future): replicates `train_claim_extractor.py` functionality behind interface.  
4. **Evaluation Stage**
   - Loads checkpoint (HF `save_pretrained` or Tinker adapter) and SciFact dev files.  
   - Computes metrics (CLAIM[c1] exact match, evidence semantic match, relation stats), writes JSONL of predictions, emits summary to stdout.  
5. **Run Metadata**
   - Collects: config path, git commit (if available), dataset SHA256, backend, training args digest, metric summary, timestamps.  
   - Writes to `runs/thinker/<timestamp>.json`.

---

## 5. Configuration Schema

### 5.1 Pipeline Config (YAML)

```yaml
tests:
  path: ../cns-support-models/tests
  markers: "not slow"
  args: ["-v"]
  enabled: true

data_validation:
  path: ../cns-support-models/data/processed/scifact_claim_extractor.jsonl
  schema:
    - name: prompt
      type: string
      required: true
      allow_empty: false
    - name: completion
      type: string
    - name: metadata
      type: object
      required: false
  max_examples: 2000

training:
  backend: hf_peft
  config_path: lora_config.yaml

evaluation:
  base_model: "meta-llama/Llama-3.1-8B-Instruct"
  checkpoint_dir: ../cns-support-models/checkpoints/claim-extractor-scifact
  claims_file: ../cns-support-models/data/raw/scifact/claims_dev.jsonl
  corpus_file: ../cns-support-models/data/raw/scifact/corpus.jsonl
  max_samples: 50
  output_path: ../../runs/thinker_eval/scifact_dev_eval.jsonl
```

### 5.2 LoRA Config (HF & Tinker Shared)

```yaml
model:
  name: "meta-llama/Llama-3.1-8B-Instruct"
  quantization: "4bit"
  device_map: "auto"
  hf_token: null  # optional

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  bias: "none"

training:
  per_device_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2e-4
  num_epochs: 3
  warmup_steps: 100
  max_seq_length: 2048
  claim_c1_weight: 5.0
  evidence_weight: 2.0
  optim: "adamw_torch"
  weight_decay: 0.01
  lr_scheduler_type: "cosine"
  fp16: true
  gradient_checkpointing: false

data:
  train_file: ../../cns-support-models/data/processed/scifact_claim_extractor.jsonl
  validation_split: 0.1
  shuffle: true
  seed: 42

output:
  checkpoint_dir: ../../cns-support-models/checkpoints/claim-extractor-scifact
  logging_steps: 10
  eval_steps: 100
  save_steps: 200
  save_total_limit: 3
  report_to: "none"
```

---

## 6. Component Design

### 6.1 Config Loader (`thinker.config`)
- Reads YAML, resolves relative paths relative to config file.  
- Validates required keys; raises descriptive errors for missing sections.  
- Converts schema definitions into `SchemaField` dataclasses (type, required, allow_empty).  
- Emits normalized dataclasses used throughout pipeline (immutable/frozen).

### 6.2 Test Runner (`thinker.validation.TestSuiteRunner`)
- Accepts `TestSuiteConfig`.  
- Invokes pytest programmatically with `pytest.main(args)` to avoid subprocess complexity.  
- Supports optional markers, additional pytest args.  
- Captures exit code; non-zero codes raise `RuntimeError`.

### 6.3 Dataset Validator (`thinker.validation.DatasetValidator`)
- Streams JSONL file; optionally caps at `max_examples`.  
- Applies schema checks (existence, type, emptiness).  
- Aggregates errors with line numbers; returns `DatasetValidationResult`.  
- CLI prints first N errors for readability; entire list available in result object.

### 6.4 Training Abstraction (`thinker.training`)

#### 6.4.1 Interface
```python
class TrainingBackend(Protocol):
    def train(self) -> TrainingReport: ...

@dataclass
class TrainingReport:
    checkpoint_dir: Path
    metrics: Dict[str, float]
    backend: str
```

#### 6.4.2 HF PEFT Backend (MVP)
- Loads LoRA config YAML via `_load_yaml`.  
- Resolves dataset + output directories relative to config file.  
- Constructs AutoModel + BitsAndBytes config if `quantization == "4bit"`.  
- Applies `prepare_model_for_kbit_training`.  
- Applies LoRA via `LoraConfig` + `get_peft_model`.  
- Loads dataset via `datasets.load_dataset("json", ...)`.  
- Splits validation set via `train_test_split`.  
- Defines tokenizer function masking prompt tokens and applying future weighted-loss support (loss masks).  
- Configures `TrainingArguments` with evaluation strategy derived from presence of eval dataset.  
- Runs `Trainer.train()`.  
- Saves checkpoint + tokenizer to `output.checkpoint_dir`.  
- Returns metrics summary (loss, eval_loss, etc.).

#### 6.4.3 Future: Tinker Backend
- Wrap existing `train_claim_extractor.py` logic but using Thinker config + run metadata.  
- Extract dataset path, LoRA hyperparameters, loss weights from shared YAML.  
- Submit `forward_backward` + `optim_step` calls via Tinker `TrainingClient`.  
- Mirror metadata logging (step counts, final loss).  
- Export adapter snapshot name corresponding to `output.checkpoint_dir`.

### 6.5 Evaluation (`thinker.evaluation`)
- Loads base model (HF) and PEFT adapter from checkpoint directory OR remote Tinker sampling client (future).  
- Loads SciFact claims (`claims_dev.jsonl`) and corpus (`corpus.jsonl`).  
- For each claim (max N), constructs prompt (hypothesis + cited docs) matching training format.  
- Generates completion with deterministic sampling defaults (temperature=0.7, top_p=0.9, do_sample).  
- Parses CLAIM[...] lines via `thinker.claim_schema`.  
- Metrics:
  - `total`: number of samples evaluated.  
  - `c1_exact_match`: count of `CLAIM[c1]` matching gold hypothesis exactly.  
  - `c1_exact_match_rate`.  
  - `evidence_semantic_match`: list of match rates per sample (string match baseline).  
  - `evidence_semantic_match_avg`.  
  - (Future) relation accuracy.  
- Writes per-sample results JSONL (claim_id, prompt, completion).  
- Returns metrics dict for CLI display + metadata.

### 6.6 CLI (`thinker.cli`)
- Command: `thinker [--config pipeline.yaml] <subcommand>`.  
- Subcommands:
  1. `validate` – run pytest + dataset validation.  
  2. `train [--backend ...] [--skip-validation]`.  
  3. `eval [--skip-validation]`.  
  4. `run [--backend ...]` – full pipeline (validate → train → eval).  
- Exit codes: 0 success; non-zero indicates failure stage.  
- Logs errors with `[thinker] error: ...`.

### 6.7 Run Metadata
- After each stage, Thinker writes structured JSON to `runs/thinker/<timestamp>.json`:
```json
{
  "timestamp": "2025-11-10T18:05:42Z",
  "config": "thinker/configs/pipeline_scifact.yaml",
  "git_commit": "abc123",
  "validation": {"pytest": "passed", "dataset_errors": 0},
  "training": {
    "backend": "hf_peft",
    "config": "thinker/configs/lora_config.yaml",
    "checkpoint_dir": "cns-support-models/checkpoints/claim-extractor-scifact",
    "metrics": {"loss": 1.04, "eval_loss": 0.93}
  },
  "evaluation": {
    "metrics": {
      "total": 50,
      "c1_exact_match_rate": 0.98,
      "evidence_semantic_match_avg": 0.62
    },
    "output": "runs/thinker_eval/scifact_dev_eval.jsonl"
  }
}
```

---

## 7. Failure Modes & Safeguards

| Stage | Failure Example | Handling |
| ----- | --------------- | -------- |
| Config | Path does not exist | Abort with clear message (`config_path.parent / raw`). |
| Tests | Pytest failure (failing unit test) | CLI prints pytest output; pipeline halts. |
| Data Validation | Missing field, invalid JSON | Collect first N errors, raise `ValueError`. |
| HF Training | Missing CUDA, low VRAM, HF token error | Propagate original exception; mention backend and config. |
| Eval | Checkpoint missing | Abort with context; suggest running `thinker train`. |
| Metadata | Git command fails | Store `git_commit: null` (non-fatal). |

---

## 8. Security & Secrets

- HF backend may require `HF_TOKEN`: Thinker should read from environment or optional config key.  
- Tinker backend requires `TINKER_API_KEY` environment variable; Thinker must fail fast if not set when backend=tinker.  
- Config files should not embed secrets; rely on env vars.  
- CLI should not print tokens (mask environment on error logs).

---

## 9. Extensibility Hooks

1. **Additional Tests:** Pipeline config can point to alternative pytest directories/markers (e.g., property tests).  
2. **Dataset Validators:** Extend schema definitions with custom validators (regex, numeric ranges).  
3. **Backends:** Add new backend factories (e.g., `hf_accelerate`, `tinker_v2`, `elixir_orchestrator`).  
4. **Metrics:** Evaluation stage can accept plugin registry to compute additional metrics (BLEU, BERTScore).  
5. **Artifacts:** Run metadata writer can emit extra files (plots, confusion matrices).

---

## 10. Implementation Backlog

1. Complete HF backend (MVP).  
2. Port Tinker backend behind Thinker interface.  
3. Integrate existing CNS pytest suites + fixtures; ensure `thinker validate` runs them.  
4. Enhance dataset validator with Schema DSL (regex, numeric ranges).  
5. Add run metadata writer + log directory management.  
6. Build CI job invoking `thinker validate` + `thinker run --backend=hf_peft --skip-validation` on sample data.  
7. Document user guide: installing dependencies, configuring pipeline, customizing stages.  
8. Long-term: integrate with Elixir-based orchestrator (“Snakepit”) per `brainstorm/20251109/COMPREHENSIVE_RESEARCH_SYNTHESIS.md`.

---

## 11. Open Questions

1. **Loss weighting parity:** HF backend currently masks prompts but does not apply token-level weights (C1/Evidence). Plan: emit `loss_mask` and implement custom Trainer (see guide).  
2. **Tinker checkpoint path mapping:** Need convention for syncing remote adapter names with local directories.  
3. **Evaluation semantics:** Should evidence semantic match use embedding similarity instead of exact match? (Spec currently uses exact string).  
4. **Large dataset splits:** How to stream > memory JSONL? (Plan: use HF `load_dataset` streaming).  
5. **Cross-platform support:** Windows vs Linux path normalization? (Currently resolved via `Path.resolve()`).

---

## 12. Appendices

- **Appendix A:** Pipeline config JSON schema (future).  
- **Appendix B:** Example CLI sessions / expected output.  
- **Appendix C:** Glossary (CLAIM[c1], SNO, QLoRA, etc.).

