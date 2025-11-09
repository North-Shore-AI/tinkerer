<div align="center">
  <img src="assets/tinkerer.svg" alt="Tinkerer Logo" width="600"/>
</div>

---

# Tinkerer Workspace

Exploratory **Chiral Narrative Synthesis (CNS)** research workspace implementing practical components of a formal framework for automated knowledge synthesis from conflicting information sources. This repository bridges theoretical formulations (documented in `cns3/`) with experimental implementations using **Thinking Machines' Tinker platform**.

## Theoretical Foundations

The `cns3/` directory contains **speculative theoretical proposals** establishing the mathematical and conceptual foundations for CNS 3.0:

- **Formal framework**: Structured Narrative Objects (SNOs), dialectical synthesis operators, multi-component critic architectures
- **Mathematical foundations**: Information geometry (Fisher metrics), algebraic topology (Betti numbers), convergence theorems
- **Architectural specifications**: Graph-based reasoning, evidence-preservation constraints, bias amplification bounds

These documents serve as **design specifications and research proposals** from which practical implementations are derived. See [`cns3/cns3_gpt5.md`](cns3/cns3_gpt5.md) and [`cns3/cns3_gemini_deepResearch.md`](cns3/cns3_gemini_deepResearch.md) for comprehensive theoretical treatments, with foundations extending from [CNS 2.0](cns2/ChiralNarrativeSynthesis_20250617.tex).

## Repository Layout

```
README.md                # Workspace overview
cns2/                    # CNS 2.0 foundational specifications (LaTeX)
cns3/                    # CNS 3.0 theoretical proposals and research designs
cns-support-models/      # Practical implementations: claim extractors, critics
notes/                   # Research journals documenting experimental iterations
```

## Quickstart (2025‑11‑09)

1. **Env setup**
   ```bash
   cd <repo>
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt  # includes tinker + helpers
   export TINKER_API_KEY=sk_live_xxx     # real key from Thinking Machines
   ```
2. **Download SciFact + preprocess**
   ```bash
   cd cns-support-models
   make scifact
   ```
3. **Train (LoRA on Llama‑3.1‑8B-Instruct)**
   ```bash
   export CNS_CLAIM_C1_WEIGHT=5.0          # force literal CLAIM[c1]
   export CNS_CLAIM_EVIDENCE_WEIGHT=2.0    # optional: weight evidence lines
   python scripts/train_claim_extractor.py --config configs/claim_extractor_scifact.yaml
   ```
4. **Evaluate**
   ```bash
   python scripts/eval_scifact_dev.py \
     --config configs/claim_extractor_scifact.yaml \
     --max-samples 50 \
     --include-gold-claim \
     --enforce-gold-claim \
     --output runs/scifact_dev_eval_canonical.jsonl
   ```
   Evaluation logs structured claims/relations, raw vs enforced metrics, and semantic matches. Use `notes/claim_extractor.md` to record each sweep.
5. **Inference sample (single prompt)**
   ```bash
   python scripts/eval_claim_extractor.py \
     --adapter-name claim-extractor-scifact \
     --prompt-file data/samples/eval_prompt.txt \
     --force-c1-text "Gold hypothesis text..."
   ```

`runs/` stays `.gitignore`d—store artifacts locally, log interesting runs in `notes/` or GitHub issues.

## Research Strategy: Theory-to-Implementation Pipeline

This workspace operationalizes a **staged implementation strategy** for the CNS 3.0 framework:

### Current Phase: Component Validation

**Tinker-Compatible Components** (present focus):
  - **Neural critics**: LoRA-adapted language models for claim extraction, grounding verification, and novelty assessment
  - **Structured evaluation**: Empirical validation of individual pipeline components against benchmark datasets
  - **Iterative refinement**: Rapid prototyping of prompt templates, loss functions, and constraint enforcement mechanisms
  - **Instrumentation**: Comprehensive telemetry capturing training dynamics and inference characteristics

### Future Phases: Full Framework Realization

**Post-Tinker Requirements** (theoretical specifications from `cns3/`):
  - **Geometric computation**: Fisher Information Metric calculation on hypothesis manifolds, geodesic distance measurement
  - **Topological analysis**: Persistent homology computation for reasoning graph coherence (Betti number extraction)
  - **Graph neural architectures**: GAT-based Logic Critics operating on typed reasoning graphs with typed edge relations
  - **Synthesis operators**: Evidence-preserving dialectical generation with formal convergence guarantees
  - **Deterministic inference**: Batch-invariant sampling with explicit RNG control for reproducibility requirements

**Implementation trajectory:** Current work validates the feasibility of neural components (critics, extractors) using accessible infrastructure. Subsequent phases will require custom implementations of the geometric and topological machinery specified in the theoretical documents, informed by empirical findings from these initial experiments.

## Collaboration & Open Questions

This workspace represents **early-stage exploratory research** implementing components of a broader theoretical framework. Collaboration is particularly valuable given the unsolved challenges at the intersection of formal methods and neural implementations.

### Current Research Questions

1. **Semantic evidence alignment**: Bridging symbolic claim structures with dense retrieval for evidence grounding
2. **Relation verification**: Validating typed edges in reasoning graphs (entailment, contradiction, support) using neural critics
3. **Constraint enforcement**: Implementing evidence-preservation guarantees during LLM-based synthesis
4. **Evaluation methodology**: Developing metrics that capture both local correctness (claim accuracy) and global coherence (reasoning graph topology)

### Engagement Protocol

When seeking feedback or reporting findings:
- **Context**: Reference theoretical specifications from `cns3/` that inform the implementation approach
- **Documentation**: Cite relevant entries in `notes/claim_extractor.md` documenting experimental iterations
- **Specificity**: Frame questions around concrete implementation challenges (e.g., "9% semantic alignment rate on SciFact validation set")
- **Theoretical grounding**: Connect empirical observations to theoretical desiderata (e.g., information preservation, bias bounds)

For Thinking Machines Discord engagement, provide: (1) repository link, (2) latest milestone summary, (3) specific technical blocker with reference to theoretical requirements.

## License

Apache 2.0
