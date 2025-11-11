# CNS Data & Validation Workflow

**Revision:** 2025-11-10  
**Purpose:** Make the complete SciFact → dataset → validation flow reproducible and documented.

---

## 1. Prerequisites
- Python 3.12+, `python3 -m venv .venv && . .venv/bin/activate`
- Install project dependencies (minimum for data/TDD loop):
  ```bash
  pip install -r requirements.txt  # if available
  pip install -e .                 # thinker's package if using editable install
  pip install pytest hypothesis
  ```
- For embedding validation install CPU-friendly Torch + SentenceTransformers:
  ```bash
  pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
  pip install sentence-transformers
  ```
- Hugging Face token only required for training/eval, **not** for data download.

---

## 2. Fetch SciFact Data
All paths below are relative to repo root (`.../tinkerer/`).

Option A (new helper):
```bash
python -m thinker.cli data setup \
  --claims-path cns-support-models/data/raw/scifact/claims_train.jsonl \
  --corpus-path cns-support-models/data/raw/scifact/corpus.jsonl \
  --output cns-support-models/data/processed/scifact_claim_extractor.jsonl
```

FEVER (same helper):
```bash
python -m thinker.cli data setup --dataset fever --skip-validation
```
This downloads FEVER claims/wiki via `scripts/download_fever.sh` (Zenodo mirrors) and runs `scripts/convert_fever.py`. Defaults expect `train.jsonl` and `wiki-pages/wiki-pages/*.jsonl` under `cns-support-models/data/raw/fever/`. Use `--fever-claims`, `--fever-wiki-dir`, `--fever-output` to override paths or include NEI claims.

Option B (legacy Make target):
```bash
cd cns-support-models
make scifact
```

`thinker data setup`/`make scifact` runs:
1. `scripts/download_scifact.sh` (downloads and extracts official `claims_*.jsonl` + `corpus.jsonl` into `data/raw/scifact/`).  
2. `scripts/convert_scifact.py` (converts to `data/processed/scifact_claim_extractor.jsonl`).

Artifacts after success:
- `cns-support-models/data/raw/scifact/claims_train.jsonl`
- `cns-support-models/data/raw/scifact/claims_dev.jsonl`
- `cns-support-models/data/raw/scifact/corpus.jsonl`
- `cns-support-models/data/processed/scifact_claim_extractor.jsonl`

Troubleshooting:
- Missing `wget`/`curl`: install via package manager.  
- Re-run `make scifact` if download interrupted (target cleans partially downloaded files).
  FEVER downloads now run automatically through the helper; if the upstream site blocks automation, download the Zenodo files manually (place `train.jsonl` and the `wiki-pages/wiki-pages/` directory under `cns-support-models/data/raw/fever/`) and rerun with `--dataset fever`.

---

## 3. Run Dataset Validator

### 3.1 Exact-Match Validation (fast default)
```bash
cd /path/to/tinkerer
python cns-support-models/scripts/validate_dataset.py \
  cns-support-models/data/processed/scifact_claim_extractor.jsonl \
  --claims-json cns-support-models/data/raw/scifact/claims_train.jsonl
```

Output:
- `... passed validation (all examples checked)` on success.  
- Descriptive error lines on failure (line numbers + issue).

### 3.2 Embedding-Based Evidence Validation (slower, semantic)
Requires corpus file and SentenceTransformers.

```bash
python cns-support-models/scripts/validate_dataset.py \
  cns-support-models/data/processed/scifact_claim_extractor.jsonl \
  --claims-json cns-support-models/data/raw/scifact/claims_train.jsonl \
  --corpus-json cns-support-models/data/raw/scifact/corpus.jsonl \
  --evidence-mode embedding \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --similarity-threshold 0.8
```

Notes:
- Script now checks for missing files and prints actionable messages (no stack traces).  
- First run downloads the embedding model (~100 MB). Cache is under `~/.cache/torch/sentence_transformers`.  
- Adjust `--similarity-threshold` depending on tolerance (0.7–0.85 typical). Higher thresholds may flag paraphrased evidence; the validator reports the offending line so you can inspect or lower the threshold.

---

## 4. Thinker Validation Loop

Thinker wraps pytest + dataset validation. Sample config already references the sample dataset, but for full dataset:

1. Update `thinker/configs/pipeline_scifact.yaml` `data_validation.path` to the processed file (`../../cns-support-models/data/processed/scifact_claim_extractor.jsonl`).  
2. Run:
   ```bash
   python -m thinker.cli --config thinker/configs/pipeline_scifact.yaml validate
   ```

This runs:
1. CNS pytest suite (`cns-support-models/tests`).  
2. Dataset validator (exact mode unless config updated).  
3. Fails fast if any prerequisite missing or test fails.

To enable embedding-based validation in Thinker, update the `data_validation` block in the pipeline config:
```yaml
data_validation:
  path: ../../cns-support-models/data/processed/scifact_claim_extractor.jsonl
  claims_json: ../../cns-support-models/data/raw/scifact/claims_train.jsonl
  corpus_json: ../../cns-support-models/data/raw/scifact/corpus.jsonl
  evidence_mode: embedding        # schema/exact/embedding
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  similarity_threshold: 0.75
  schema:
    - name: prompt
      type: string
    - name: completion
      type: string
```
Setting `evidence_mode: schema` skips the external script for fast iterations.

---

## 5. Optional: Generate Mini Fixtures

For CI / quick local tests, fixtures live under `cns-support-models/tests/fixtures/`. To regenerate:
```bash
python cns-support-models/scripts/convert_scifact.py \
  --claims_json cns-support-models/tests/fixtures/scifact_claims_sample.jsonl \
  --corpus_json cns-support-models/tests/fixtures/scifact_corpus_sample.jsonl \
  --out cns-support-models/tests/fixtures/scifact_dataset_sample.jsonl
```

These fixtures are already committed; regenerate only if schema changes.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
| ------- | ----- | ----|
| `Dataset JSONL not found...` | Path typo or dataset not generated | Run `make scifact` or confirm path |
| `sentence-transformers must be installed...` | Embedding mode without dependency | Install CPU torch + `sentence-transformers` as described above |
| `embedding mode requires --claims-json and --corpus-json` | Missing flags | Provide both files (download via `make scifact`) |
| Pytest failures in `thinker validate` | Schema or converter regression | Inspect line numbers, fix converter, re-run tests |

---

## 7. Caching SentenceTransformers / HF Artifacts
- Set `export SENTENCE_TRANSFORMERS_HOME=$PWD/.cache/sentence-transformers` (or similar) to keep embeddings local to the repo/CI runner.  
- Alternatively set `HF_HOME` or `TRANSFORMERS_CACHE` to reuse Hugging Face caches across runs:
  ```bash
  export HF_HOME=$HOME/.cache/huggingface
  export SENTENCE_TRANSFORMERS_HOME=$HF_HOME/sentence-transformers
  ```
- In CI, cache the directory referenced above between jobs to avoid repeated downloads.

## 8. Future Enhancements
- Handle authenticated/download-token flows if FEVER mirrors change again.  
- Auto-generate dataset-specific pipeline configs so `thinker data setup` can drop in ready-to-run YAMLs.
