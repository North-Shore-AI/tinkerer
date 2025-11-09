Chiral Narrative Synthesis Support Models: Scientific & Technical Proposal

———

### 1. Executive Summary

- Problem: Existing RAG/debate systems cannot transform conflicting evidence into structured, verifiable
  narratives. CNS support models must (i) extract atomic claims with logical relations and (ii) provide
  dialectical synthesis guarantees.
- Solution: LoRA fine‑tuned Llama‑3.1 adapters trained via Tinker on SciFact (and upcoming FEVER)
  with structured prompts/outputs (CLAIM/RELATION graphs). Training code (cns-support-models/scripts/
  train_claim_extractor.py:1‑195) implements pipelined forward_backward/optim_step calls, weighted CE loss,
  and adapter checkpoints. Evaluation scripts enforce schema and run structured dev sweeps.
- Outcomes: Claim F1 ≥0.82 on SciFact dev, relation accuracy ≥0.78, schema adherence 99%, FEVER generalization
  target F1≥0.70. Deliverables include dataset converters, configs, Tinker loops, eval harnesses, and
  reproducible logs.
- Resources: Compute provided by Tinker preview; local CPU for preprocessing. Estimated 400 GPU-hours for
  ablations + FEVER training.

———

### 2. Background & Motivation

- Context: Dialectical synthesis requires graph-structured representations of claims (CNS §4.1). Current
  extractors (GPT-4 zero-shot, T5 fine-tunes) lack formal schemas, making critics unusable.
- Related Work: RAG frameworks treat conflicts as noise; argument mining lacks scaling to multi-domain
  corpora. LoRA-based adapters (Hu et al. 2021) show low-rank adaptation suffices for domain-specific tasks.
- Motivation: Without rigorous extraction, CNS critics cannot enforce Theorems on convergence/bias. SciFact
  proves feasibility; FEVER adds domain breadth (news, policy), enabling subsequent dialectical experiments.

———

### 3. Technical Approach

#### 3.1 Problem Definition

- Input: Document $d \in \Sigma^{*}$.
- Output: Claim graph $G = (C, E, \lambda)$ with $C={c_i}$, $E \subseteq C \times C$, $\lambda:E \to
    {\text{supports},\text{refutes},\text{contrasts}}$.
- Constraints: Atomicity (single proposition per $c_i$), grounding (each $c_i$ maps to span in $d$), logical
    consistency.
- Success Criteria: Claim precision/recall ≥0.80, relation accuracy ≥0.75, schema compliance >0.99.

#### 3.2 Architecture (code refs)

- LoRA Parameterization (train_claim_extractor.py:60‑78):
    $$W = W_0 + \frac{\alpha}{r}BA,; B \in \mathbb{R}^{m \times r}, A \in \mathbb{R}^{r \times n}, r=16,
    \alpha=32$$
- Tokenizer workflow (train_claim_extractor.py:60‑78):
  - Prompt tokens weight=0; completion weight=1 (prevents loss on input).
- Data pipeline: convert_scifact.py:22‑145 and convert_fever.py:1‑130 create structured JSONL.

#### 3.3 Training Procedure

- Loss: Weighted cross-entropy (prompt tokens masked) (train_claim_extractor.py:109‑137).
- Optimizer: Adam (lr=1.5e‑4, clip=1.0). Pipelined Tinker futures ensure same clock cycle.
- Config: configs/claim_extractor_scifact.yaml + claim_extractor_fever.yaml.

#### 3.4 Evaluation

- Interactive: scripts/eval_claim_extractor.py enforces schema in prompts, prints completions.
- Structured: scripts/eval_scifact_dev.py loads dev set, samples, parses CLAIM/RELATION, reports match rate.

———

### 4. Experimental Design

#### 4.1 Datasets

- SciFact: Biomedical claims (train 505 after filtering). Conversion script builds prompts with abstract context.
- FEVER: General-domain claims. Manual download → convert_fever.py produces 10k+ JSONL entries (supports/refutes). Wiki sentences hashed for evidence.

#### 4.2 Baselines

- GPT-4 zero-shot (format drift).
- T5-Large fine-tuned (domain-locked).
- Proposed LoRA adapter (schema-adherent, multi-domain).

#### 4.3 Ablations (planned)

- LoRA rank (4/8/16/32).
- Loss weighting (relation-heavy vs uniform).
- Dataset mixing (SciFact only vs SciFact+FEVER).

———

### 5. Theoretical Analysis

- Generalization: PAC-Bayes argument—low-rank updates reduce KL divergence, improving bounds.
- Rate-Distortion: Claim graphs minimize rate (few claims) subject to distortion (semantic coverage).
- Complexity: Training/inference remain $O(Ld^2)$ dominated by base, LoRA adds $O(Lrd)$ overhead.

———

### 6. Reproducibility

- Repo: cns-support-models/.
- Scripts: training, evaluation, converters with documented configs.
- Env: .venv created via uv, packages (torch CPU, transformers, tinker).
- Logs/checkpoints: adapters saved via Tinker; evaluation outputs saved to JSONL.

———

### 7. Broader Impacts

- Enables CNS 3.0 critics/synthesis by providing structured inputs.
- Supports multi-domain claim extraction (science + news).
- Risks: misuse for misinformation detection without human oversight; mitigated via human validation.

———

