# Antagonist MVP RFC — Post-Proposer Semantic/Topology Instrumentation

**Date:** 2025‑11‑18  
**Authors:** CNS Support Models Team  
**Status:** Draft for implementation  

## 0. Motivation

The latest Proposer evaluation (adapter `claim-extractor-scifact-20251118T173307`) delivered full semantic metrics plus the new topology instrumentation (`logic/betti.py`, `metrics/chirality.py`). Headline figures:

- Schema compliance **100%**, citation accuracy **96%**.
- Mean entailment **0.448** (38% ≥0.75), mean semantic similarity **0.25** (20% ≥0.70), overall semantic pass **38%**.
- β₁ **0** across all 50 SciFact dev samples (graphs are DAGs).
- Mean chirality score **0.561**, mean Fisher-Rao distance **16.75** (healthy tension even without cycles).

This profile means Antagonist can no longer rely on finding cycles; instead, it must target polarity contradictions and evidence asymmetries surfaced by the chirality metric. This RFC converts those insights into concrete requirements for the Antagonist MVP.

## 1. Inputs & Interfaces

| Source | Payload | Notes |
| --- | --- | --- |
| `runs/thinker_eval/<run>.jsonl` | Per-claim record with `completion`, semantic validation payload, `beta1`, `cycles`, `chirality.{score,fisher_rao,evidence_overlap,polarity_conflict}` | Produced by `thinker/evaluation.py` after every run (now with per-sample logging). |
| Proposer SNO manifests | `snos.jsonl`, `manifest.json` | Same as Section 3.1 of AGENTS.md; Antagonist should enrich `manifest.json` with chirality deltas. |
| Config thresholds | `antagonist.yaml` (new) | Defines chirality trigger (default ≥0.55), polarity_conflict trigger, max retrieval fan-out, critic weights. |

## 2. Functional Requirements

1. **Polarity Stress Tests (primary objective)**  
   - Trigger when `chirality.score ≥ 0.55` or `chirality.polarity_conflict == True`.  
   - Retrieve counter-evidence by inverting claim polarity (regex negation + embedding anti-neighbors).  
   - Emit structured flags with `issue_type=POLARITY_CONTRADICTION`, evidence snippets, and entailment/conflict scores.

2. **Evidence Consistency Check**  
   - Use citation IDs + `semantic_validation.entailment_score`.  
   - Flags when cited evidence is neutral/contradictory (`entailment_score < 0.5`) even if β₁=0.  
   - Stores `critic_score=1 - entailment_score`.

3. **Residual β₁ Monitoring (fallback)**  
   - Even though current β₁=0, keep the betti hook alive.  
   - When future datasets produce β₁>0, Antagonist reuses the same interface to generate cycle-busting countergraphs.

4. **Outputs**  
   - `artifacts/antagonist/<run>/flags.jsonl` with fields: `claim_id`, `issue_type`, `chirality_delta`, `evidence`, `critic_scores {entailment, chirality, beta1}`, `severity`.  
   - Updated manifest linking counter-SNOs or retrieval evidence to `claim_id`.

## 3. Architecture

```
Proposer SNO ──▶ Thinker Eval (semantic + topology) ──▶ Antagonist Runner
                                                   │             │
                                                   └── metrics ──┘
```

- **Runner implementation**: Python module `thinker/antagonist.py` (new) invoked via `python -m thinker.cli antagonist --config ...`.  
- **Retrieval primitive**: reuse the existing sentence-transformer embedder to search for anti-neighbors; fall back to regex negation for quick wins.  
- **Critic cascade**: Schema → Grounding → Logic already run upstream. Antagonist consumes their outputs; any new critic should append scores to the same JSONL schema to stay consistent with `Evaluation Agent` expectations in AGENTS §2.3.

## 4. Metrics & Telemetry

| Metric | Definition | Target | Notes |
| --- | --- | --- | --- |
| Precision | TRUE flags / issued flags | ≥0.8 | Measured against synthetic contradiction suite. |
| Recall | TRUE flags / known contradictions | ≥0.7 | Same suite; adjust thresholds using chirality score percentiles. |
| β₁ estimation error | | ≤10% | Compare Antagonist’s betti calculation against `logic/betti.py` outputs (now available per sample). |
| Chirality delta coverage | Share of high-chirality cases receiving a flag | ≥0.9 | Because β₁=0, chirality is the main signal; coverage must be near-total. |

Telemetry: log every flag with `chirality.score`, `evidence_overlap`, `polarity_conflict`, retrieval metadata, and critic scores to `runs/antagonist_eval/*.json`.

## 5. Implementation Plan

1. **Week 1**  
   - Land `thinker/antagonist.py` CLI entry point + config schema.  
   - Build polarity heuristic pass (regex negation + chirality threshold).  
   - Write unit tests using synthetic SNO JSONL fixtures.
2. **Week 2**  
   - Integrate embedding anti-neighbor search + DeBERTa contradiction scoring.  
   - Emit structured flags + manifest updates.  
   - Add CLI command to Thinker (`thinker.cli antagonist`).
3. **Week 3**  
   - Hook into CI: `thinker.cli antagonist --config ...` after Proposer eval runs.  
   - Wire metrics collection to comparison dashboard.  
   - Iterate thresholds using latest `runs/thinker_eval` artifacts.

## 6. Open Questions

1. Should Antagonist block downstream synthesis when chirality stays high but evidence overlap is low? (Proposal: log as MED severity, allow Synthesizer to decide.)  
2. How many retrieval candidates per flag are acceptable before latency becomes an issue? Need benchmark once embedding anti-search lands.  
3. Do we require human review before escalating `issue_type=POLARITY_CONTRADICTION`? Proposed: only for severity ≥HIGH, triggered when `chirality≥0.7` and evidence overlap ≥0.5.

---

**Next actions:**  
- Implement the runner + CLI.  
- Convert `runs/thinker_eval/scifact_dev_eval.jsonl` into a regression set for contradiction heuristics.  
- Update AGENTS.md once Antagonist MVP ships (precision/recall targets + telemetry evidence).
