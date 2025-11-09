<div align="center">
  <img src="assets/tinkerer.svg" alt="Tinkerer Logo" width="600"/>
</div>

---

# Tinkerer Workspace

Exploratory **Chiral Narrative Synthesis (CNS)** research built on **Thinking Machines’ Tinker platform**. Expect rapid iteration; each subdirectory carries its own README for reproducing experiments.

## Repository Layout

```
README.md                # Workspace overview
cns-support-models/      # Claim-extractor training & evaluation scaffolding
cns3/                    # CNS 3.0 concept papers, critiques, and proposals
notes/                   # Research journals (see notes/claim_extractor.md)
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

## Direction & Tinker Fit

- **What Tinker is great for (today):**
  - LoRA-based adapters on Llama-class models (claim extractor, dialectical synthesizer, critics).
  - Managed training/sampling infrastructure, telemetry, structured evals.
  - Multi-stage pipelines via scripts (train → sample → feed onward).
  - Rapid iteration without touching GPU infrastructure.

- **Where Tinker stops (tomorrow):**
  - Custom architectures/manifold math from the CNS 3.0 papers (Betti numbers, Fisher metrics, etc.).
  - Strict batch-invariant inference (their API doesn’t expose per-row RNG controls—see “Defeating Nondeterminism in LLM Inference” on their blog).
  - Embedded multi-agent runtimes (everything still goes through training/sampling endpoints).
  - Non-LoRA losses that require redefining the base model internals.

**Plan:** use Tinker to de-risk the LoRA adapters, logging + structured evals as we go. Once we need bespoke statistical manifolds or hard inference guarantees, we’ll need deeper hooks or custom infrastructure built on the learnings here.

## Sharing & Collaboration

- Reference the relevant entry in `notes/claim_extractor.md` when opening an issue or asking for feedback (e.g., in the Thinking Machines Discord).
- Highlight open questions—currently semantic evidence alignment and relation verification.
- Be explicit that the project is experimental; collaborators are welcome precisely because the hard parts (evidence alignment, multi-agent synthesis) are unsolved.

When posting in the Tinker Discord Projects room, include:
1. Link to this repo.
2. Summary of the latest milestone (canonical hypotheses, structured eval).
3. The blocker you want help with (e.g., only 9% semantic hits).

## License

Apache 2.0
