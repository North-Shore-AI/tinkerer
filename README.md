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

Git-tracked paths include `README.md`, `docs/`, `CNS_PROPOSAL.md`, `cns2/`, `cns3/`, `cns-support-models/`, `assets/`, `LICENSE`, and `repomix.config.json`. Local-only directories (`runs/`, `brainstorm/`, interim `data/` outputs, `tinker-docs/`, `thinking-machines-labs/`, legacy `cns/`) are ignored to keep the public presence sharply curated. When sharing findings, never assume ignored paths are available to reviewers—summaries and artifacts must live in the tracked structure above.

## Repository Guide

| Path | Description |
| --- | --- |
| `README.md` | This overview plus operational guidance for reviewers and collaborators. |
| `docs/RESEARCH_PORTFOLIO.md` | Orientation guide detailing artifacts, public-scope policy, and enhancement log. |
| `CNS_PROPOSAL.md` | Executive summary for the CNS support-models plan with numbered sections for proposal committees. |
| `cns2/` | Historical CNS 2.0 LaTeX specification anchoring the theoretical lineage. |
| `cns3/` | CNS 3.0 theoretical documents (geometry/topology framing, validation memos, revised proposals). |
| `cns-support-models/` | Implementation scaffold: configs, scripts, Makefile, experiment logs (`notes/claim_extractor.md`). |
| `assets/` | Branding assets referenced by public docs. |

## Operational Quickstart (2025‑11‑09)

1. **Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install tinker pyyaml
   export TINKER_API_KEY=sk_live_xxx   # Thinking Machines credential
   ```
2. **Data (SciFact)**
   ```bash
   cd cns-support-models
   make scifact
   ```
3. **Train (LoRA on Llama‑3.1‑8B-Instruct)**
   ```bash
   export CNS_CLAIM_C1_WEIGHT=5.0
   export CNS_CLAIM_EVIDENCE_WEIGHT=2.0   # optional evidence emphasis
   python scripts/train_claim_extractor.py --config configs/claim_extractor_scifact.yaml
   ```
4. **Evaluate (structured SciFact sweep)**
   ```bash
   python scripts/eval_scifact_dev.py \
     --config configs/claim_extractor_scifact.yaml \
     --max-samples 50 \
     --include-gold-claim \
     --enforce-gold-claim \
     --output runs/scifact_dev_eval_canonical.jsonl
   ```
   Log each sweep in `cns-support-models/notes/claim_extractor.md`—this is the canonical experiment ledger.
5. **Interactive sampling**
   ```bash
   python scripts/eval_claim_extractor.py \
     --adapter-name claim-extractor-scifact \
     --prompt-file data/samples/eval_prompt.txt \
     --force-c1-text "Gold hypothesis text..."
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
- Link to `CNS_PROPOSAL.md` for executive context when engaging program committees or funding partners.

## License

Apache 2.0
