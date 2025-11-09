<div align="center">
  <img src="assets/tinkerer.svg" alt="Tinkerer Logo" width="600"/>
</div>

---

# Tinkerer Workspace

Exploratory CNS (Chiral Narrative Synthesis) research built on **Thinking Machines' Tinker platform**. Expect rapid iteration; each subdirectory carries its own README for reproducing experiments.

## Repository Layout

```
README.md                # Workspace overview
cns-support-models/      # Claim-extractor training & evaluation scaffolding
cns3/                    # CNS 3.0 concept papers, critiques, and proposals
notes/                   # Research journals (see notes/claim_extractor.md)
```

## Current Highlights (2025‑11‑09)

- `notes/claim_extractor.md` documents the latest milestone: canonical `CLAIM[c1]` enforcement, structured SciFact telemetry, semantic/relational gaps, and next actions.
- `cns-support-models/README.md` explains the Tinker setup, links to the CNS 3.0 papers in `cns3/`, and lists the exact evaluation command (`--include-gold-claim --enforce-gold-claim`).
- Post-processing now reanchors every non‑`c1` claim to its closest passage sentence (see `scripts/eval_scifact_dev.py` and `scripts/claim_schema.py`), guaranteeing that downstream CNS components ingest grounded evidence.
- `runs/` is intentionally `.gitignore`d; persist important observations in `notes/` or GitHub issues instead of committing bulky artifacts.

## Contributing & Sharing

The CNS concept papers are intentionally speculative. If you share or extend this repo:

1. Reference the relevant research-log entry so collaborators understand the project’s maturity.
2. Highlight the open engineering questions—currently semantic alignment of evidence claims and relation verification.
3. Emphasize that the work is experimental and feedback/collaboration are welcome.

Please open an issue or start a thread in the Thinking Machines research channel (Tinker Discord) with links back to the run notes when proposing changes.***
