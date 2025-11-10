# CNS Support Models: Scientific & Technical Proposal

Concise summary of the implementation workstream that underpins **Chiral Narrative Synthesis (CNS) 3.0** on Thinking Machines' Tinker platform.

---

## 1. Executive Summary

| Dimension | Details |
| --- | --- |
| **Problem** | Retrieval-Augmented Generation (RAG) and debate-style systems fail to turn conflicting sources into verifiable Structured Narrative Objects (SNOs). CNS requires schema-faithful claim graphs plus dialectical synthesis guarantees. |
| **Proposed solution** | Fine-tune LoRA adapters on Llama‑3.1 via Tinker using structured CLAIM/RELATION prompts. Training code (`cns-support-models/scripts/train_claim_extractor.py`) pipelines `forward_backward` and `optim_step`, applies weighted cross-entropy, and checkpoints adapters. Evaluation scripts enforce schemas and run SciFact/FEVER sweeps. |
| **Outcomes (targets)** | SciFact dev Claim F1 ≥ 0.82, relation accuracy ≥ 0.78, schema adherence ≥ 0.99. FEVER generalization goal Claim F1 ≥ 0.70. Deliverables include dataset converters, configs, LoRA loops, evaluation harnesses, and reproducible logs. |
| **Resources** | Tinker preview compute; local CPU for preprocessing. Estimated 400 GPU-hours to cover SciFact ablations + FEVER training. |

## 2. Background & Motivation

Dialectical synthesis described in `cns3/` demands graph-structured representations with provable properties (CNS §4.1). Existing extractors (GPT-4 zero-shot, T5 fine-tunes) drift from formal schemas, preventing critic enforcement of the convergence, information-preservation, and bias bounds promised theoretically. SciFact provides a biomedical proving ground; FEVER adds cross-domain coverage needed before scaling to CNS critics and synthesis experiments.

## 3. Technical Approach

### 3.1 Problem Definition

- **Input**: document \(d \in \Sigma^{*}\).
- **Output**: claim graph \(G=(C,E,\lambda)\) with claims \(C=\{c_i\}\), edges \(E\subseteq C\times C\), and relation labels \(\lambda:E\to\{\text{supports},\text{refutes},\text{contrasts}\}\).
- **Constraints**: atomic claims, grounded spans, and logical consistency enforced by schema parsers.
- **Success metrics**: claim precision/recall ≥ 0.80, relation accuracy ≥ 0.75, schema compliance ≥ 0.99.

### 3.2 Architecture

- **LoRA parameterization** (`train_claim_extractor.py`):
  \[
  W = W_0 + \frac{\alpha}{r} BA,\quad B \in \mathbb{R}^{m \times r},\ A \in \mathbb{R}^{r \times n},\ r=16,\ \alpha=32
  \]
- **Tokenizer workflow**: prompt tokens weight 0, completion tokens weight 1+ to avoid input loss.
- **Data pipeline**: `scripts/convert_scifact.py` and `scripts/convert_fever.py` produce structured JSONL (prompt/completion pairs) with traceable metadata.

### 3.3 Training Procedure

- **Loss**: weighted cross-entropy with prompt masking and claim-specific weights (CLAIM[c1] emphasis + optional evidence weighting via `CNS_CLAIM_EVIDENCE_WEIGHT`).
- **Optimizer**: Adam (`learning_rate=1.5e-4`, gradient clipping 1.0). `forward_backward` and `optim_step` futures are pipelined to share a clock cycle.
- **Configuration**: YAML manifests in `cns-support-models/configs/` capture datasets, adapter names, and logging cadence. Runs emit structured provenance JSON to `runs/`.

### 3.4 Evaluation

- **Interactive**: `scripts/eval_claim_extractor.py` samples adapters with schema enforcement switches (`--force-c1-text`, `--force-c1-file`).
- **Structured**: `scripts/eval_scifact_dev.py` rebuilds prompts, enforces CLAIM/RELATION schema, and reports literal/semantic matches plus relation accuracy. Outputs land in `runs/*.jsonl`.

## 4. Experimental Design

### 4.1 Datasets

- **SciFact**: 505 filtered biomedical claims with aligned abstracts; Makefile target `make scifact` downloads raw JSONL and produces `data/processed/scifact_claim_extractor.jsonl`.
- **FEVER**: 10k+ general-domain claims; manual download followed by `scripts/convert_fever.py` to build multi-evidence JSONL with hashed Wikipedia spans.

### 4.2 Baselines

1. GPT‑4 zero-shot (schema drift).
2. T5-Large supervised fine-tune (domain-locked).
3. Proposed LoRA adapter (schema-adherent, multi-domain) – target solution.

### 4.3 Planned Ablations

- LoRA rank sweep: \(r \in \{4,8,16,32\}\).
- Loss weighting modes: uniform, CLAIM[c1]-heavy, evidence-heavy.
- Dataset mixing: SciFact only vs. SciFact + FEVER blended curricula.

## 5. Theoretical Analysis

- **Generalization**: PAC-Bayes framing—low-rank updates reduce KL divergence between posterior/prior, aligning with CNS bias bounds.
- **Rate-Distortion**: Claim graphs minimize rate (claim count) under semantic distortion constraints, mirroring information-preservation requirements in CNS.
- **Complexity**: Training/inference dominated by base model \(O(Ld^2)\); LoRA adds \(O(L r d)\), keeping adapter deployment lightweight.

## 6. Reproducibility & Deliverables

- **Code**: `cns-support-models/` houses Makefile, YAML configs, dataset converters, LoRA training loop, and evaluation utilities.
- **Environment**: Python 3.10+, virtual environment recommended; dependencies include `tinker`, `pyyaml`, and standard library modules.
- **Logging**: `train_claim_extractor.py` writes provenance metadata (git commit, dataset hash, adapter name, loss curve) to timestamped JSON. Evaluation harnesses output structured `.jsonl` for downstream analysis.
- **Documentation**: Experiment iterations recorded in `cns-support-models/notes/claim_extractor.md`.

## 7. Broader Impacts & Risk

- **Positive impact**: Supplies the structured inputs required for CNS critics and synthesis, enabling verifiable reasoning over scientific and policy corpora.
- **Dual-use considerations**: Same models could be repurposed for surveillance or misinformation flagging without human oversight. Mitigation: enforce human-in-the-loop validation and publish schema documentation alongside any release.

--- 
