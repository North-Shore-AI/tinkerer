Thinker — TDD‑First Pipelines for CNS 3.0 Claim Extraction

Overview
- Thinker is a small, config‑driven orchestrator that enforces a “test‑before‑GPU” workflow for CNS 3.0 support models. It wires together: dataset prep, schema and semantic validation, LoRA training, and evaluation against CNS 3.0’s semantic metrics.
- It aligns with the CNS Agent Playbook (Proposer/Antagonist/Synthesizer) and implements the operational pieces needed before the full dialectical loop is online.

Key Features
- Test gate: runs pytest and dataset validation before training.
- Data bootstrap: helpers for SciFact and FEVER conversion/validation.
- Training backends: local HF/PEFT (QLoRA) and Tinker (remote service) with manifest hand‑off.
- Evaluation: 4‑stage semantic validation (Citation → Entailment → Similarity → Paraphrase) with legacy exact‑match for debugging.
- Schema helpers for CLAIM[...] and RELATION lines used by the claim extractor.
- Single YAML pipeline config controls validation, training, and evaluation.

Repository Layout (selected)
- `cli.py`: CLI entry points (validate/train/eval/run, data setup).
- `pipeline.py`: end‑to‑end orchestration and state hand‑off.
- `config.py`: typed config models and YAML loader.
- `validation.py`: pytest runner and JSONL dataset validator.
- `semantic_validation.py`: CNS 4‑stage semantic validator.
- `evaluation.py`: evaluation harness + completion providers (HF/PEFT or Tinker).
- `training.py`: HF/PEFT trainer and Tinker script runner.
- `claim_schema.py`: parsers for CLAIM[...] and RELATION lines.
- `configs/`: example pipeline and LoRA configs.
- `tests/`: minimal coverage for CLI, config, pipeline, validation, training.

Prerequisites
- Python 3.10+
- For HF/PEFT training or evaluation
  - `torch`, `transformers`, `peft`, `datasets`, `accelerate`, `sentence-transformers`, `bitsandbytes`, `pyyaml`, `pytest`.
  - NVIDIA GPU recommended. 4‑bit quantization uses `bitsandbytes`.
- For Tinker backend (optional)
  - `tinker` Python package and a valid `TINKER_API_KEY` environment variable.
- Data conversion scripts live in the sibling repository `cns-support-models/` (expected at repo root: `../cns-support-models`).

Install
- From the repository root (so Python can import `thinker`):
  - `pip install -U torch transformers peft datasets accelerate sentence-transformers bitsandbytes pyyaml pytest`
  - Optional: `pip install tinker` (for Tinker sampling/training)
  - You can run without installing as a package by invoking `python -m thinker.cli` from the repository root.

Quick Start (Debug)
- Use the smaller debug configs to smoke‑test locally.
- Data (SciFact, minimal):
  - `python -m thinker.cli data setup --dataset scifact --skip-validation`
    - This calls `make scifact` in `../cns-support-models` and writes a processed JSONL.
- Validate:
  - `python -m thinker.cli --config configs/pipeline_scifact_debug.yaml validate`
- Train (HF/PEFT, QLoRA):
  - `python -m thinker.cli --config configs/pipeline_scifact_debug.yaml train --backend hf_peft`
- Evaluate (choose one):
  - HF/PEFT: copy `configs/pipeline_scifact_debug.yaml`, set `evaluation.backend: hf_peft`, set `evaluation.base_model` and `evaluation.checkpoint_dir` to the HF checkpoint, then run:
    - `python -m thinker.cli --config <your_eval_cfg>.yaml eval`
  - Tinker: ensure a Tinker adapter manifest exists at `runs/latest_tinker_adapter.json` and set `evaluation.backend: tinker` (default in debug). Export `TINKER_API_KEY` and run:
    - `python -m thinker.cli --config configs/pipeline_scifact_debug.yaml eval`
- End‑to‑End:
  - `python -m thinker.cli --config configs/pipeline_scifact_debug.yaml run`

CLI Overview
- `thinker info`
  - Prints Thinker/Tinker versions, Python/platform info, and the active pipeline configuration (tests, validation, training, evaluation sections).
- `thinker manifest [--path custom_manifest.json]`
  - Shows adapter metadata (name, path, base model, timestamp) from the resolved manifest file (defaults to `evaluation.tinker_manifest_path` or `runs/latest_tinker_adapter.json`).
- `thinker validate`
  - Runs pytest (if enabled in config) and dataset schema/semantic validation.
- `thinker train [--backend hf_peft|tinker] [--skip-validation]`
  - Trains using the configured backend.
- `thinker eval [--skip-validation]`
  - Evaluates the latest checkpoint/adapter against CNS semantic metrics.
- `thinker run [--backend ...]`
  - Convenience: validate → train → eval.
- `thinker data setup [options]`
  - Prepares SciFact or FEVER data and (optionally) runs an external validator.

Helper script: run `./thinker.sh` from the repo root to bootstrap the virtualenv, install dependencies, and access a menu for data setup, validation, training/evaluation (HF or Tinker), diagnostics (`info`, `manifest`), and debug runs.

Data Setup
- SciFact
  - `python -m thinker.cli data setup --dataset scifact [--validation-mode exact|embedding] [--skip-validation]`
  - Uses `make scifact` in `../cns-support-models`, copies `data/processed/scifact_claim_extractor.jsonl`, and (unless skipped) runs `scripts/validate_dataset.py` (exact or embedding) which also writes a filtered `data/processed/scifact_claim_extractor_clean.jsonl`.
- FEVER
  - `python -m thinker.cli data setup --dataset fever [--fever-include-nei]`
  - Downloads raw FEVER (uses `scripts/download_fever.sh`) or expects you to place files under `cns-support-models/data/raw/fever`, then converts via `scripts/convert_fever.py`.
  - If the official site requires manual access, download FEVER manually and rerun the command.

Configuration
- Pipeline YAML (`configs/pipeline_scifact.yaml` and `configs/pipeline_scifact_debug.yaml`)
  - `tests`: pytest location, markers, extra args, `enabled` flag.
  - `data_validation`: JSONL path, schema (`prompt`, `completion`, optional `metadata`), `max_examples`, `evidence_mode` (`schema|exact|embedding`), embedding model and similarity threshold, optional `claims_json`/`corpus_json` for external validation, and relation gating (`relation_field`, `require_relations`).
  - `training`: `config_path` for LoRA YAML, backend (`hf_peft` or `tinker`), optional Tinker script/config path and log dir.
  - `evaluation`: backend (`hf_peft` or `tinker`), `base_model`, `checkpoint_dir` (HF), or Tinker adapter info (`tinker_manifest_path`, `tinker_adapter_*`), output path, sampling params.
- LoRA YAML (`configs/lora_config.yaml`, `configs/lora_config_debug.yaml`)
  - `model`: HF model name, optional `quantization: 4bit`, `device_map: auto`.
  - `lora`: standard PEFT hyperparameters (`r`, `alpha`, `dropout`, `target_modules`, `bias`).
  - `training`: batch size, accumulation, lr, epochs, warmup, max seq length, scheduler, weight decay, fp16.
  - `data`: `train_file` (JSONL prompt/completion), optional `validation_split`, `seed`.
  - `output`: `checkpoint_dir`, logging/eval/save steps, `save_total_limit`, `report_to`.

Evaluation and Metrics
- CNS 3.0 rejects strict string equality as a success criterion. Thinker reports semantic‑first metrics and includes exact‑match only as a debugging probe.
- 4‑Stage Semantic Validation (from AGENTS.md §4.1; implemented in `semantic_validation.py`)
  - Stage 1 — Citation Accuracy (hard gate): generated output must cite the gold evidence doc IDs.
  - Stage 2 — Entailment: DeBERTa‑v3 NLI entailment score ≥ 0.75 between cited evidence (premise) and the generated claim (hypothesis).
  - Stage 3 — Semantic Similarity: sentence‑transformers cosine similarity ≥ 0.7 to the gold claim.
  - Stage 4 — Paraphrase Tolerance: if 1–2 pass, near‑misses in 3 can still be accepted.
- Topology + Chirality diagnostics log β₁, detected cycles, Fisher‑Rao distance, and chirality scores for each evaluation row (`logic/betti.py`, `metrics/chirality.py`).
- Outputs
  - Per‑example validation results are written to `evaluation.output_path` (JSONL), and aggregate metrics are printed to stdout.
  - Legacy exact‑match metrics are retained for side‑by‑side comparison.

Backends
- HF/PEFT (`training.backend: hf_peft`, `evaluation.backend: hf_peft`)
  - Produces a local checkpoint dir; evaluation loads the base model and applies the LoRA adapter for generation.
  - Set `evaluation.base_model` and `evaluation.checkpoint_dir`.
- Tinker (`training.backend: tinker`, `evaluation.backend: tinker`)
  - Runs the legacy Tinker training script and writes `runs/latest_tinker_adapter.json` with adapter metadata (name, path, base model).
  - Export `TINKER_API_KEY` and install the `tinker` package.
  - Evaluation uses the manifest or explicit `evaluation.tinker_adapter_*` fields.

Schema: CLAIM and RELATION
- Thinker expects completions in a simple, parseable format:
  - `CLAIM[c1]: <main hypothesis>`
  - `CLAIM[c#] (Document <doc_id>): <supporting/refuting claim>`
  - `RELATION: <source_id> <supports|refutes|contrasts> <target_id>`
- Parsers live in `claim_schema.py` and are used during evaluation to extract `CLAIM[c1]` and referenced evidence.
- Graph debugging helper: `python scripts/build_graph.py --dataset <jsonl> --line 1` prints nodes, β₁, and detected cycles for a sample completion.

Typical Workflows
- “Validate then train” (recommended)
  - `python -m thinker.cli --config configs/pipeline_scifact.yaml validate`
  - `python -m thinker.cli --config configs/pipeline_scifact.yaml train`
- Single‑shot (validate → train → eval)
  - `python -m thinker.cli --config configs/pipeline_scifact.yaml run`
- Evaluate a trained artifact
  - HF: set `evaluation.backend: hf_peft` + `base_model` + `checkpoint_dir` → `python -m thinker.cli --config ... eval`
  - Tinker: ensure manifest present or set explicit adapter path → `python -m thinker.cli --config ... eval`

Outputs and Artifacts
- HF/PEFT checkpoints: as configured in the LoRA YAML `output.checkpoint_dir`.
- Tinker adapter manifest: `runs/latest_tinker_adapter.json` (by default when using the Tinker trainer).
- Evaluation results: `evaluation.output_path` (JSONL under `runs/...` in example configs).

Troubleshooting
- “validate_dataset.py not found; cannot run external validation”
  - Ensure the sibling repository `cns-support-models/` exists at the expected path and includes `scripts/validate_dataset.py`.
- “No Tinker adapter information available …”
  - Provide `evaluation.tinker_manifest_path`, or set `evaluation.tinker_adapter_name` and `evaluation.tinker_adapter_path`, or run Tinker training to produce `runs/latest_tinker_adapter.json`.
- “TINKER_API_KEY is not set …”
  - Export a valid key: `export TINKER_API_KEY=...` before using the Tinker backend.
- CUDA/4‑bit issues
  - Install a compatible `bitsandbytes` and CUDA toolchain, or switch to CPU/FP16 and adjust LoRA config.
- Exact‑match is low while semantic scores are good
  - Expected. Exact‑match is a debugging probe per CNS 3.0; rely on the semantic metrics for success criteria.

Testing
- Run the built‑in tests from repo root:
  - `pytest thinker/tests -q`
- The default pipeline validates a small SciFact sample before allowing training to proceed.

Alignment with CNS Agent Playbook
- Proposer: Thinker trains and evaluates claim‑extraction models that emit SNO scaffolds (CLAIM/RELATION with citations).
- Antagonist: Future integration will add contradiction heuristics and β₁/chirality deltas into Thinker’s evaluation stage.
- Synthesizer: Thinker prepares manifests and evaluation hooks intended to plug into a constrained decoding + critic loop.
- Evaluation policy: semantic‑first metrics align with AGENTS.md §1.0/§4.1 (exact‑match is a debugging probe only).

File References
- CLI: `cli.py:1`
- Pipeline: `pipeline.py:1`
- Config models: `config.py:1`
- Validation: `validation.py:1`
- Semantic validation: `semantic_validation.py:1`
- Evaluation harness: `evaluation.py:1`
- Training backends: `training.py:1`
- Claim schema parsers: `claim_schema.py:1`
- Example pipeline configs: `configs/pipeline_scifact.yaml:1`, `configs/pipeline_scifact_debug.yaml:1`
- Example LoRA configs: `configs/lora_config.yaml:1`, `configs/lora_config_debug.yaml:1`

License
- Internal project; see repository root for licensing details.
