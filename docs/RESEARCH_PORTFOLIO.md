# Chiral Narrative Synthesis Research Portfolio

This document orients reviewers at Thinking Machines to the public-facing scope of the **Tinkerer** workspace, highlights the major artifacts that accompany the repository, and records the professionalism-focused enhancements applied in this revision cycle.

## 1. Public Scope & Git Hygiene

The `.gitignore` policy keeps sensitive or high-churn assets (e.g., `runs/`, `brainstorm/`, `tinker-docs/`, `thinking-machines-labs/`, historical `cns/`) private, while exposing the long-lived research artifacts listed below. Only the tracked paths are considered part of the public academic record.

| Category | Paths | Notes |
| --- | --- | --- |
| **Tracked** | `README.md`, `docs/`, `CNS_PROPOSAL.md`, `cns-support-models/`, `cns3/`, `cns2/`, `assets/`, `LICENSE`, `repomix.config.json` | These files comprise the formal research portfolio and implementation scaffold. |
| **Ignored / local only** | `runs/`, `brainstorm/`, `tinker-docs/`, `thinking-machines-labs/`, historical `cns/`, intermediate data directories (`data/`, `outputs/`, `figures/`, etc.) | Use for experiments, brainstorming, and proprietary data; do not rely on them for reproducible context. |

## 2. Directory & Artifact Guide

| Path | Role | Primary Audience | Status |
| --- | --- | --- | --- |
| `README.md` | High-level overview, quickstart, and collaboration guide | External reviewers & new collaborators | Revamped (Nov 2025) |
| `docs/RESEARCH_PORTFOLIO.md` | Orientation, scope clarifications, enhancement log | Thinking Machines academic reviewers | New (Nov 2025) |
| `CNS_PROPOSAL.md` | Concise executive & technical summary of CNS support models | Program committees, funding boards | Reformatted (Nov 2025) |
| `cns2/ChiralNarrativeSynthesis_20250617.tex` | CNS 2.0 foundational LaTeX specification | Researchers studying historical theory | Unmodified |
| `cns3/` | CNS 3.0 theoretical documents (dialectical geometry, validation memos, research proposals) | Theory group & reviewers | Active development |
| `cns-support-models/` | Practical implementation scaffold (configs, scripts, notes) | Engineering collaborators | Active development |
| `cns-support-models/notes/claim_extractor.md` | Experiment log for the claim extractor | Experiment owners | Updated regularly |
| `assets/tinkerer.svg` | Branding asset for README | Comms | Stable |

## 3. Research Tracks

### 3.1 Theoretical & Proposal Track (`cns2/`, `cns3/`)

- **CNS 2.0 Specification (`cns2/ChiralNarrativeSynthesis_20250617.tex`)** – Formalizes Structured Narrative Objects (SNOs), critic architectures, and dialectical synthesis guarantees. Serves as the theoretical baseline.
- **CNS 3.0 Portfolio (`cns3/`)** – Five public artifacts covering revised proposals, mathematical foundations, validation memos, and deep research reports:
  - `CNS_3_0_A_DIALECTICAL_FRAMEWORK_FOR_AUTOMATED_KNOWLEDGE_DISCOVERY.md` – Flagship proposal describing SNO-3, critic ensembles, and scaling targets.
  - `cns3_gpt5.md` – Abstract + registered experimental plan transitioning CNS from blueprint to implementation.
  - `cns3_gemini_deepResearch.md` – Extended literature review and research framing with benchmark strategy.
  - `20251109_revised_cns_proposal_for_thinking_machines.md` – Updated pitch emphasizing the topological-geometric perspective (CNS-TGM).
  - `20251109_technicalValidation_CNSSupportModelsScientificProposal.md` – Independent validation + risk analysis for the support-model plan.

### 3.2 Implementation & Validation Track (`cns-support-models/`)

- `configs/` – YAML descriptors for SciFact/FEVER fine-tunes, explicitly mapping datasets, adapters, and optimization knobs.
- `scripts/train_claim_extractor.py` – Tinker LoRA training loop with weighted loss, provenance logging, and adapter export.
- `scripts/eval_*.py` – Schema-aware evaluation harnesses (interactive and structured SciFact sweeps).
- `scripts/convert_*.py` & `data/samples/` – Reproducible dataset preprocessing utilities and smoke-test artifacts.
- `notes/claim_extractor.md` – Run-by-run log documenting enforcement changes, metrics, and next steps.

## 4. Professionalization Enhancements

| Status | Enhancement | Notes |
| --- | --- | --- |
| Complete | Documented public scope and repository orientation (`docs/RESEARCH_PORTFOLIO.md`) | Clarifies what reviewers can rely on and how ignored directories map to local workflows. |
| Complete | Refreshed `README.md` with accurate layout, professional tone, and Thinking Machines engagement guidance | Reconciles quickstart instructions with existing files and links to orientation materials. |
| Complete | Reformatted `CNS_PROPOSAL.md` for executive readability (headings, numbered sections, cross-references) | Aligns with grant/proposal conventions and surfaces key metrics. |
| Planned | Add dependency manifest + reproducible environment recipe for `cns-support-models/` | Will enumerate Python packages and pin versions once upstream APIs are finalized. |
| Planned | Expand automated validation (lint/tests) for schema parsers and evaluation scripts | Targeting lightweight pytest suite referencing `data/samples/`. |

