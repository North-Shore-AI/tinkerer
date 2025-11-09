# CNS Support Models

This directory contains the scaffolding for the auxiliary models that power Chiral Narrative Synthesis (CNS) 3.0. CNS needs several LoRA-tunable LLM components—in particular, (1) a *claim & hypothesis extraction* model that turns raw documents into Structured Narrative Objects (SNOs) and (2) a *dialectical synthesis* model that rewrites thesis/antithesis pairs into grounded syntheses. Both are excellent fits for Tinker, which lets us write lightweight Python loops while it handles distributed training.

## Directory layout

```
cns-support-models/
├── README.md                # This file
├── configs/                 # YAML configs describing datasets, models, and LoRA hyperparameters
├── data/
│   └── samples/             # Toy data artifacts for quick smoke tests
└── scripts/                 # Training & utility scripts that call into the Tinker API
```

## Initial scope

1. **Claim & hypothesis extractor** – fine-tunes a smaller Llama-3.1 adapter to emit atomic claims plus relations (supports/refutes). This feeds the CNS ingestion pipeline described in `cns3/cns3_gemini_deepResearch.md`.
2. **Dialectical synthesizer** – upgrades the base Llama-3.1 synthesis engine with task-specific prompts, constrained decoding, and preference loss signals.

We are starting with (1) because CNS §4.1 calls out Llama-3.1-8B as the preferred claim extractor, and its training loop is straightforward to express via Tinker.

## How to use the scaffold

1. Drop domain-specific training data into `data/<dataset>/` (JSONL with `prompt`, `completion`, and optional metadata). A tiny example lives in `data/samples/claim_extractor.jsonl`.
2. If you want ready-made corpora, run `make scifact` (see below) for SciFact. For FEVER, download the Zenodo files manually (see `data/fever/README.md`), run `python scripts/convert_fever.py`, then point at `configs/claim_extractor_fever.yaml`.
3. Run `python scripts/train_claim_extractor.py --config configs/claim_extractor.yaml`. The script:
   - Loads the config and sample data schema
   - Creates a `tinker.ServiceClient` and `TrainingClient`
   - Streams mini-batches through `forward_backward` / `optim_step` in a pipelined fashion so they land in the same clock cycle
   - Periodically saves weights for downstream evaluation / deployment

Future additions will include:

- A shared data schema module
- Evaluation scripts that call `sample()` for regression tests
- Preference / DPO pipelines for the dialectical synthesizer

## Requirements

- Python 3.10+
- `tinker` and `tinker-cookbook` packages installed (see `tinker-docs/install.md`)
- `TINKER_API_KEY` exported in your shell environment

## Evaluation

- Quick spot-check: `python scripts/eval_claim_extractor.py --adapter-name claim-extractor-scifact`
- Structured SciFact dev sweep: `python scripts/eval_scifact_dev.py --config configs/claim_extractor_scifact.yaml --max-samples 50 --output runs/scifact_dev_eval.jsonl`

## Dataset quickstart (SciFact)

We standardized on **SciFact** as the initial supervised dataset because it already couples scientific claims with supporting/refuting evidence, mirroring the CNS ingestion flow. To download and preprocess it:

```bash
cd cns-support-models
make scifact
```

This will (1) download the upstream tarball into `data/raw/scifact/` and (2) run `scripts/convert_scifact.py` to produce `data/processed/scifact_claim_extractor.jsonl`. Train with:

```bash
python scripts/train_claim_extractor.py --config configs/claim_extractor_scifact.yaml
```

Use `make clean` to remove the local copy if needed.

## Next steps

- Flesh out the data loaders to cover multi-turn conversations
- Integrate the critics' outputs as weak labels for targeted fine-tuning
- Add preference-based objectives for the dialectical synthesizer once the claim extractor loop is solid
