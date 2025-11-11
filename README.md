<div align="center">
  <img src="assets/tinkerer.svg" alt="Tinkerer Logo" width="600"/>
</div>

---

# Tinkerer Workspace · Chiral Narrative Synthesis

Exploratory **Chiral Narrative Synthesis (CNS)** program maintained for **Thinking Machines** reviewers. The repository captures both the formal theory (CNS 2.0 → CNS 3.0) and the practical **Tinker-based** implementation track for Structured Narrative Objects (SNOs), critics, and evaluation harnesses.

## Purpose & Audience

- **Audience**: academics and applied scientists evaluating CNS readiness for collaborative investment.
- **Scope**: public artifacts only; sensitive experiments stay in ignored directories per the `.gitignore`.
- **Orientation**: see [`docs/RESEARCH_PORTFOLIO.md`](docs/RESEARCH_PORTFOLIO.md) for a catalog of artifacts, research tracks, and professionalism enhancements made in this revision.

## Public Scope

Git-tracked paths include `README.md`, `docs/`, `docs/CNS_PROPOSAL.md`, `cns2/`, `cns3/`, `cns-support-models/`, `assets/`, `LICENSE`, and `repomix.config.json`. Local-only directories (`runs/`, `brainstorm/`, interim `data/` outputs, `tinker-docs/`, `thinking-machines-labs/`, legacy `cns/`) are ignored to keep the public presence sharply curated. When sharing findings, never assume ignored paths are available to reviewers—summaries and artifacts must live in the tracked structure above.

## Repository Guide

| Path | Description |
| --- | --- |
| `README.md` | This overview plus operational guidance for reviewers and collaborators. |
| `docs/RESEARCH_PORTFOLIO.md` | Orientation guide detailing artifacts, public-scope policy, and enhancement log. |
| `docs/CNS_PROPOSAL.md` | Executive summary for the CNS support-models plan with numbered sections for proposal committees. |
| `cns2/` | Historical CNS 2.0 LaTeX specification anchoring the theoretical lineage. |
| `cns3/` | CNS 3.0 theoretical documents (geometry/topology framing, validation memos, revised proposals). |
| `cns-support-models/` | Implementation scaffold: configs, scripts, Makefile, experiment logs (`notes/claim_extractor.md`). |
| `assets/` | Branding assets referenced by public docs. |

## Operational Quickstart (2025‑11‑10)

1. **Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -e . pytest hypothesis
   pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
   pip install sentence-transformers  # needed for embedding validation
   ```
   Export `TINKER_API_KEY=sk_live_xxx` before running any Tinker-backed commands.
2. **Data via Thinker helper**
   - **SciFact:**
     ```bash
     python -m thinker.cli data setup --dataset scifact --validation-mode embedding --similarity-threshold 0.7
     ```
   - **FEVER (Zenodo mirrors):**
     ```bash
     python -m thinker.cli data setup --dataset fever --skip-validation
     ```
     (The helper downloads all JSONL shards + `wiki-pages.zip` automatically; if the remote host throttles you, drop the files under `cns-support-models/data/raw/fever/` and rerun.)
3. **Validation-first loop**
   ```bash
   python -m thinker.cli --config thinker/configs/pipeline_scifact.yaml validate
   ```
   This runs the CNS pytest suite plus dataset validation (exact or embedding based on the config).
4. **Training**
   ```bash
   # Local smoke run (requires GPU on this machine or a cheap GPU VM)
   python -m thinker.cli train --backend hf_peft

   # Full remote run on Tinker (requires TINKER_API_KEY)
   python -m thinker.cli train --backend tinker
   ```
   Tinker runs log provenance JSON under `runs/` and refresh `runs/latest_tinker_adapter.json` with the newest adapter name/path so downstream commands know which checkpoint to sample.
5. **Evaluation**
   ```bash
   python -m thinker.cli eval
   ```
   Evaluation now talks to Tinker directly: Thinker loads the tokenizer via the API, samples from the adapter recorded in `runs/latest_tinker_adapter.json`, and writes metrics/completions to `runs/thinker_eval/…`. No Hugging Face download is required as long as the manifest exists (created automatically by every Tinker training run). To override the adapter, set `evaluation.tinker_adapter_*` in the pipeline config or drop a custom manifest file in `runs/`.
6. **GPU options (why HF/PEFT exists)**
   - **Local smoke tests:** A single 24 GB GPU (e.g., RTX 3090/4090, RTX 6000, A5000) is enough for QLoRA. Renting one from a provider (RunPod, Lambda Labs, Vast.ai) costs ~$0.50–$1.50/hr—handy for config/dataset validation before you spend Tinker cycles.
   - **Fast iterations:** The HF/PEFT backend sticks around for cheap local debugging, but Tinker is now the default path for production training/eval. The workflow is still `thinker validate` → `thinker train --backend tinker` → `thinker eval`.
   - **Direct Tinker runs:** If you’d rather skip HF entirely, run the menu/CLI options that point at the Tinker backend; validation always happens locally first to keep remote jobs clean.

7. **Interactive scripts (legacy)**
   You can still call the original scripts directly if needed:
   ```bash
   python cns-support-models/scripts/train_claim_extractor.py --config cns-support-models/configs/claim_extractor_scifact.yaml
   python cns-support-models/scripts/eval_scifact_dev.py --config cns-support-models/configs/claim_extractor_scifact.yaml
   python cns-support-models/scripts/eval_claim_extractor.py --adapter-name claim-extractor-scifact
   ```

Use the `.gitignore`d `runs/` directory for local artifacts; only promote curated summaries into tracked notes or issues.

## Research Tracks & Status

- **Theoretical track (`cns2/`, `cns3/`)** – Documents the evolution from CNS 2.0 to CNS 3.0, including the algebraic-topological framing (`CNS_3_0_A_DIALECTICAL_FRAMEWORK_FOR_AUTOMATED_KNOWLEDGE_DISCOVERY.md`), the CNS-TGM revision (2025‑11‑09 proposal), and independent validation memos. These serve as review packages for Thinking Machines academics.
- **Implementation track (`cns-support-models/`)** – Contains the LoRA training loops, dataset converters, evaluation utilities, and experiment logs needed to operationalize the claim-extractor critic on the Tinker platform. See `docs/RESEARCH_PORTFOLIO.md` for a detailed artifact index.

## Engagement & Reporting

When requesting feedback or posting updates (e.g., Discord, review memos):
- Reference the specific theoretical artifact (`cns3/...`) or implementation log entry that motivates the question.
- Summarize new results in tracked notes before sharing (ignored directories are invisible to reviewers).
- Report blockers with concrete metrics (e.g., "Semantic alignment 9% on SciFact dev despite enforced CLAIM[c1]").
- Link to `docs/CNS_PROPOSAL.md` for executive context when engaging program committees or funding partners.

## License

Apache 2.0
