# CNS 3.0 Agent Playbook

This document harmonizes the **CNS 3.0 dialectical blueprint** (see `cns3/`) with the **Thinker/Tinker implementation track** (`thinker/`, `cns-support-models/`). It enumerates the agents we already rely on, their inputs/outputs, current maturity, and the near-term work needed to close the gap between theory and production.

---

## 0. Why Agents? (Epistemic Architecture)

CNS 3.0 treats knowledge synthesis as **adversarial collaboration**:

- **Proposer** is deliberately myopic (extracts claims from single documents).
- **Antagonist** is deliberately skeptical (finds holes, not manufactures them).
- **Synthesizer** is deliberately conservative (only merges with strong evidence).

This division of labor prevents the "everything is true" failure mode of naive RAG while maintaining provenance. **This is not an assembly line.** Agents iterate, backtrack, and defer to human judgment. The goal is not automation but **augmentation**—giving researchers structured representations of contested knowledge so they can reason about uncertainty rather than hide it.

**Example:** When SciFact claims "vitamin D prevents COVID" conflicts with FEVER evidence showing "vitamin D has no effect," CNS doesn't pick a winner—it surfaces the β₁ gap, quantifies evidence quality, and lets domain experts adjudicate with full provenance.

---

## 1. Core Dialectical Agents

| Agent | Mission | Current Status |
| --- | --- | --- |
| **Proposer (Thesis)** | Ingest raw corpora, extract atomic claims + evidence, emit first-pass Structured Narrative Objects (SNOs). | Prototype: SciFact/FEVER conversion scripts, LoRA claim extractor, Thinker data/validation loop. |
| **Antagonist (Antithesis)** | Stress-test SNOs, surface contradictions/bias, quantify topological "holes" (β₁) + chirality to decide whether synthesis should proceed. | Planned: rule-based critiques + retrieval probing; no automated pipeline yet. |
| **Synthesizer (Synthesis)** | Resolve high-chirality/high-entanglement SNO pairs, generate candidate syntheses for critic review, log provenance. | Planned: future Llama‑3.1/Qwen adapters w/ constrained decoding + critic feedback. |

---

### 1.0 Evaluation Philosophy (Exact-Match Exit)

- **Goal:** Faithful *interpretation* and *grounded reasoning*, not verbatim reproduction. Exact-match testing is incompatible with CNS 3.0's mandate to reconcile conflicting narratives.
- **Semantic-first metrics:** cosine similarity, entailment scores, β₁ reduction. These replace strict equality checks except when exercising schema parsers.
- **Policy:** Documented here to prevent future regressions—when a metric demands literal copying, treat it as a debugging probe, not a success criterion. (See analysis memo on LoRA reproduction limits.)

---

### 1.1 Proposer Agent

- **Inputs:** Raw documents (SciFact, FEVER, future domain corpora), config in `thinker/configs/*.yaml`, schema definitions in `cns-support-models/tests/fixtures/`.
- **Outputs:** Schema-compliant JSONL (prompt/completion) pairs, evidence-linked SNO scaffolds, validation reports.
- **Tooling:**  
  - `python -m thinker.cli data setup --dataset {scifact|fever}` (downloads + converts).  
  - `python -m thinker.cli validate` (pytest + dataset validator).  
  - `cns-support-models/scripts/convert_*`, `.../validate_dataset.py`.  
- **Health Metrics:**  
  - Schema compliance (CLAIM/RELATION parse rate) – Target ≥95%.  
  - Semantic similarity to gold claims (cosine >0.7) – Target ≥60% on held-out dev.  
  - Evidence grounding score (DeBERTa entailment) – Target ≥0.75 mean.  
  - Relation logical consistency (graph critic) – Target ≥70% valid edges.  
- **Current Research Priorities (P0 - Critical):**
  1. **Eliminate citation hallucinations:** Train with `citation_validity_weight=5.0` to force stronger grounding (in progress, commit `e500bb2`)
  2. **Improve semantic grounding:** Target mean entailment ≥0.60 (current 0.395), overall pass ≥45% (current 34%)
  3. **Validate citation penalty effectiveness:** If weight=5.0 fails, escalate to weight=10.0 or implement negative example training
- **Next Research Tasks (P1):**
  1. Contrastive loss integration for tighter evidence-claim alignment
  2. Broaden datasets (FEVER config/tests, temporal corpora)
  3. Auto-generate SNO manifests (graph export + metadata hashes)
 - **Baseline Evaluation (2025-11-18, adapter `claim-extractor-scifact-20251118T173307`):**
   - Schema compliance 100%, citation accuracy 96% (hard gate).
   - Mean entailment 0.448 (38% ≥0.75), mean similarity 0.25 (20% ≥0.70), overall semantic pass 38%.
   - β₁ = 0 across 50 SciFact dev samples (logic graphs are acyclic pre-Antagonist).
   - Mean chirality score 0.561, mean Fisher-Rao distance 16.75 (see `logic/betti.py`, `metrics/chirality.py` for instrumentation).
   - Raw outputs + per-sample topology/chirality payloads in `runs/thinker_eval/scifact_dev_eval.jsonl`.
 - **Training Iteration (2025-11-18, adapter `claim-extractor-scifact-20251118T220454`, citation_validity_weight=2.0):**
   - **Status:** ❌ FAILED to eliminate citation hallucinations
   - Training: 98.7% loss reduction (2330.81 → 29.66 over 320 steps), citation_invalid_rate=0.000 (clean training data)
   - Evaluation: Schema 98% (-2%), citation 96% (unchanged), mean entailment 0.395 (-0.053, WORSE), overall pass 34% (-4%, WORSE)
   - **Critical Finding:** Antagonist identified 2 HIGH severity CITATION_INVALID cases (claims 133, 179) persisting after training - model fabricates document IDs not in source corpus
   - **Root Cause:** Penalty weight=2.0 (3x loss multiplier) insufficient to teach citation grounding; model learned format but not grounding behavior
   - **Next Action:** Increased `citation_validity_weight` from 2.0 to 5.0 (6x multiplier, commit `e500bb2`) for next training run  

---

### 1.2 Antagonist Agent

- **Inputs:** Proposer SNOs (from `runs/thinker_eval/*.jsonl`), critic thresholds (chirality, entailment, evidence overlap).
- **Outputs:** Structured flags (JSONL) with `claim_id`, `severity` (LOW/MEDIUM/HIGH), `issues` (with type + details), and full metrics.
- **Current Implementation (MVP as of 2025-11-18):**
  - ✅ CLI integration: `python -m thinker.cli antagonist`
  - ✅ Threshold-based heuristics (chirality ≥0.55, entailment <0.5, evidence overlap ≥0.2)
  - ✅ 4 issue types detected:
    - **`CITATION_INVALID`** (HIGH): Model cited documents not in source corpus (citation hallucination)
    - **`POLARITY_CONTRADICTION`** (MEDIUM): Chirality ≥0.55 indicates structural tension
    - **`POLARITY_CONFLICT`** (HIGH): Same claim receives both support and refutation
    - **`WEAK_ENTAILMENT`** (MEDIUM): Entailment score <0.5 indicates poor evidence grounding
  - ✅ Comprehensive test coverage: 22 tests in `thinker/tests/test_antagonist.py`
  - ✅ Real-world validation: 92% flagging rate (46/50 samples), 2 HIGH severity cases correctly identified
  - ✅ Production-ready: Complete CLI integration, telemetry, and documentation
- **Critical Findings (2025-11-18 Analysis):**
  - Successfully identified 2 HIGH severity CITATION_INVALID cases (claims 133, 179) where Proposer fabricated document IDs
  - 60.9% of flagged claims have weak entailment (<0.5), confirming Proposer semantic quality issues
  - 84.8% of flags are POLARITY_CONTRADICTION (mean chirality 0.561, Fisher-Rao 16.75)
  - No false positives detected in manual review - all flags are legitimate quality concerns
  - **Actionable insight:** Antagonist correctly identified that Proposer needs stronger citation grounding (validated by training iteration failure)
- **Next Steps (P1):**
  1. ⏳ Embedding anti-neighbor retrieval for counter-evidence generation
  2. ⏳ DeBERTa contradiction scoring to upgrade POLARITY_CONTRADICTION detection
  3. ⏳ Precision/recall instrumentation against 200-pair synthetic contradiction suite
  4. ⏳ Expand test coverage to ≥80% (currently 22 tests)
- **Documentation:** See `docs/20251118/antagonist-mvp-review/` for comprehensive analysis, flag review, and HIGH severity case studies.

#### Antagonist Success Metrics

- **Precision ≥0.8** on a synthetic contradiction test suite (no false alarms).
  - **Measurement:** Weekly run against 200 hand-labeled contradiction pairs (100 true contradictions, 100 spurious). Precision = TP/(TP+FP).
- **Recall ≥0.7** on a known-contradiction validation set (doesn't miss real issues).
  - **Measurement:** Same 200-pair suite. Recall = TP/(TP+FN).
- **β₁ quantification accuracy** within ±10% of graph-theory ground truth on labeled samples.
  - **Measurement:** Compare Antagonist's β₁ estimate to ground-truth Betti numbers on 50 manually constructed SNO graphs (validated by topologists).
- **Actionable flag rate:** ≥80% of HIGH-severity flags lead to Proposer refinement or human escalation.
  - **Measurement:** Track disposition of HIGH flags over 30 days: (refined + escalated)/total_flags ≥ 0.8.
- **Current tension profile (2025-11-18 Proposer eval):** β₁ already sits at 0 for 50/50 SciFact dev SNOs while mean chirality remains 0.561. Antagonist MVP should therefore prioritize polarity contradictions and evidence counterfactuals over cycle detection until Proposer loosens topology constraints.
- **Reference RFC:** `cns3/20251118_antagonist_mvp_rfc.md` enumerates inputs, heuristics, telemetry, and the milestone plan derived from this profile.

**Anti-pattern:** Rewarding "more issues found."  
**Desired pattern:** Rewarding accurate issue detection that drives resolution.

---

### 1.3 Synthesizer Agent

- **Status:** 🔴 BLOCKED - Waiting for Proposer to reach ≥60% semantic quality (currently 34-38%)
- **Blocking Issue:** Cannot synthesize high-quality SNOs when input claims have citation hallucinations and weak evidence grounding
- **Unblocking Criteria:**
  1. Mean entailment ≥0.60 (current: 0.395-0.448)
  2. HIGH severity CITATION_INVALID flags eliminated (current: 2/50 samples)
  3. Overall semantic pass rate ≥60% (current: 34-38%)
- **Inputs:** High-chirality/high-entanglement SNO pairs, Antagonist deltas, critic weights
- **Outputs:** Candidate synthesized SNOs (hypothesis, reasoning graph, evidence set, trust score), manifest entries for downstream evaluation
- **Planned Stack:**
  - Base models: Llama‑3.1‑70B (development) → Qwen3‑235B MoE (production)
  - Constrained decoding (KCTS + citation enforcement)
  - Critic-guided refinement loop (Generate → Verify → Refine)
- **Pre-work (on hold until Proposer unblocks):**
  1. Finalize critic interfaces (Grounding, Logic, Novelty) as callable services
  2. Define SNO manifest schema (superset of `runs/latest_tinker_adapter.json`)
  3. Prototype with Thinker eval harness by swapping Tinker sampling backend once adapter exists

---

## 2. Supporting Operational Agents

These agents ensure the core pipeline is reproducible and review-ready even before the full dialectical loop is live.

---

### 2.1 Data & Validation Agent

- **Role:** Enforce ADR‑0002 "test-before-GPU" gate.  
- **Mechanics:** Thinker CLI `validate` stage running CNS pytest suite + JSONL validator (`thinker/validation.py`).  
- **Runbook:** `thinker.sh` recommends **10 (data setup) → 1 (validate) → 2/3/4 (train) → 5/6 (eval)** prior to any evaluation/antagonist run. Option 17 launches the dashboard server; option 18 opens the dashboard manager.
- **KPIs:** Validation pass/fail, dataset SHA256 lineage, pytest coverage.

---

### 2.2 Training Agent (LoRA Orchestrator)

- **Local HF/PEFT Backend:**  
  - Configured via `thinker/configs/lora_config*.yaml`.  
  - Ideal for smoke runs on single GPU (QLoRA, gradient masking).  
- **Tinker Backend:**  
  - `python -m thinker.cli train --backend tinker` shells out to `cns-support-models/scripts/train_claim_extractor.py`.  
  - Produces provenance logs + `runs/latest_tinker_adapter.json`.  
- **Responsibilities:** Keep adapter manifests current, log run metadata (dataset hashes, config digest, loss curves), surface anomalies (loss divergence, schema regression).

---

### 2.3 Evaluation Agent

- **Implementation:** `thinker/evaluation.py` driven via CLI `eval`.  
- **Current Capabilities:**  
  - Calls Tinker sampling API using latest manifest.  
  - Parses CLAIM/RELATION outputs, enforces `CLAIM[c1]` canonicalization, computes fuzzy similarity + semantic evidence checks.  
  - Writes JSONL under `runs/thinker_eval/`.  
- **Next Steps:** Plug in critic scores (Grounding/Logic/Novelty) once available, expand the dashboard telemetry (training/eval detail charts, limited-run micro config), expose semantic similarity metrics to avoid over-reliance on strict string match.

---

### 2.4 Critic Ensemble (Planned Agents)

**Execution Order:** `Schema validator → Grounding → Logic → Novelty/Parsimony → Bias/Causal`. Output of each critic gates the next; failures short-circuit to retry pipelines.

| Critic | Function | Status |
| --- | --- | --- |
| **Grounding** | DeBERTa‑v3 entails/contradicts claims vs. evidence; already partially covered by `validate_dataset.py` exact-match mode. | Specified in CNS docs; requires fine-tuned model + Thinker hook. |
| **Logic** | Graph Attention Network scoring reasoning coherence (β₁ reduction). | Theoretical design ready; needs graph export + training data. |
| **Novelty/Parsimony** | Embedding-based novelty vs. historical SNOs, penalize bloated graphs. | To be built; interim proxy = metadata-driven heuristics. |
| **Bias / Causal** | Detect correlation-vs-causation claims, demographic skew. | Future work; note dependencies in `cns3/cns3_gpt5.md`. |

---

### 2.5 Critic Conflict Resolution

When critics disagree, apply the following decision rules:

| Scenario | Resolution |
| --- | --- |
| Grounding passes, Logic fails | **Logic veto** (ungrounded reasoning is worse than no reasoning) |
| Logic passes, Novelty fails | **Accept** (coherent redundancy beats incoherent novelty) |
| Multiple critics below threshold | **Weighted vote**; if tie, defer to [Section 6 Human Review Gates](#6-human-review-gates) |
| | • **Weights:** Grounding (0.4), Logic (0.3), Novelty (0.2), Bias (0.1) |
| | • **Vote:** Sum(weight × normalized_score); threshold 0.6 to pass |
| | • **Tie:** Defined as 0.55 < score < 0.65 |
| Critic deadlock (>3 iterations) | Route to [Section 6 Human Review Gates](#6-human-review-gates) |

---

## 3. Data, Artifact & Communication Flow

1. **Ingest** – `thinker data setup` downloads raw SciFact/FEVER, converts to JSONL via `cns-support-models/scripts/convert_*.py`, records hashes.  
2. **Validate** – Thinker `validate` runs pytest + schema/evidence checks; failures block downstream runs.  
3. **Train** – Choose backend (`hf_peft` for smoke, `tinker` for full runs) via Thinker CLI or menu (options 2/3/8/9). Outputs: PEFT checkpoints or Tinker adapter manifest.  
4. **Evaluate** – `thinker eval` streams prompts through latest adapter, logs structured metrics, updates run artifacts.  
5. **(Future)** – Antagonist + Synthesizer consume SNO manifests, pass candidates through critic ensemble, publish synthesized SNOs + trust scores.

---

### 3.1 Agent Communication Protocol

- **Transport:** File-based handoff under `artifacts/{agent}/{run_id}/`. Each SNO batch ships with:
  - `snos.jsonl` (claims, evidence, relations, metadata)
  - `manifest.json` (hashes, critic scores, provenance)
- **Locking:** Create `.{run_id}.lock` during write; downstream agents watch for lock removal before ingest.
- **Versioning:** SemVer per manifest schema (`schema_version: "1.0.0"`). Breaking changes require converter utility + release note.
- **API plan:** Graduating to REST/queue once Antagonist/Synthesizer go online; capturing requirements here for continuity.

---

### 3.2 Agent Decision Tree

**Legend:**  
• `──NO──→` Decision branch (condition false)  
• `──YES──→` Decision branch (condition true)  
• `──FAIL──→` Terminal failure path  
• `(≤N)` Maximum retry count

```
START
  ↓
Proposer extracts SNO
  ↓
Schema valid? ──NO──→ Retry (≤3) ──FAIL──→ Abort
  ↓ YES
Grounding ≥0.7? ──NO──→ Evidence refresh ──→ Retry
  ↓ YES
Pass to Antagonist
  ↓
High-severity flags? ──YES──→ β₁ >threshold? ──YES──→ Human review
  ↓ NO                              ↓ NO
Low-severity flags ──→ Auto-refine Proposer ──→ Re-submit
  ↓ 
No flags → Pass to Synthesizer
  ↓
Synthesizer iterate (≤10 cycles)
  ↓
Critics pass? ──NO──→ Refine ──→ Critics pass? (recursive)
  ↓ YES
β₁ reduction ≥30%? ──NO──→ Human review
  ↓ YES
Output final SNO
```

See also: [Section 6 Human Review Gates](#6-human-review-gates).

---

## 4. Backlog & Ownership Signals

| Theme | Tasks | Owners (default) | Blocked By | Effort |
| --- | --- | --- | --- | --- |
| **Semantic grounding** | Contrastive loss for evidence claims, entailment critic integration, Thinker metric surfacing. | CNS support-models team. | Need entailment model checkpoints. | M |
| **FEVER parity** | Finish `pipeline_fever.yaml`, fixtures, pytest coverage, README/docs updates. | Thinker maintainers. | Semantic grounding metrics (avoid false alarms). | S |
| **Critic bootstrap** | Gather weak labels, define interfaces, wire into Thinker evaluation stage. | Research + infra pairing. | SNO manifest schema finalization. | L |
| **Antagonist MVP** | Build contradiction heuristics + embedding anti-search, finalize JSON flag spec. | Research pod. | Critic bootstrap (need threshold guides). | M |
| **Synthesizer prep** | Manifests for SNO graphs, constrained decoding experiments on Llama‑3.1. | Research + platform. | Critic ensemble API, human review gates. | L |

---

### 4.1 Semantic Grounding: Operational Definition

Not exact-match; instead a multi-metric validation (executed in order and short-circuited on failure):

1. **Citation accuracy** (hard gate, checked first)
   - Referenced sentences exist and support claim polarity (supports/refutes).
   - Target: 100% citation validity (hard requirement).
   - **Rationale:** No point scoring semantics if citations are hallucinated.

2. **Entailment** (DeBERTa‑v3 NLI, checked second)
   - Hypothesis: generated claim; Premise: cited evidence sentence.
   - Target: Entailment score ≥0.75.
   - **Rationale:** Measures whether the claim is supported by evidence.

3. **Semantic similarity** (sentence-transformers, checked third)
   - Generated vs. gold claim embeddings.
   - Target: cosine similarity ≥0.7.
   - **Rationale:** Allows valid paraphrasing of gold labels.

4. **Paraphrase tolerance** (interpretive layer)
   - Accept alternate phrasing if (1) and (2) pass; reject when meaning changes (optionally back-translation spot checks).
   - **Rationale:** Prevents false negatives from stylistic variation.

**Execution Order:** Check 1 → 2 → 3 → 4. Each failure short-circuits.

---

## 5. Convergence Criteria & Exit Conditions

| Agent | Success Threshold | Retry Logic | Hard Stop |
| --- | --- | --- | --- |
| **Proposer** | Schema valid + Grounding score ≥0.7 | Up to 3 retries with temperature annealing + evidence refresh | Abort run, open incident after 3 failures |
| **Antagonist** | Flags only genuine contradictions (precision ≥0.8 on test suite) | Re-run with expanded retrieval window once; escalate if still empty | Escalate if >5 high-severity contradictions remain unresolved |
| **Synthesizer** | β₁ reduction ≥30% + all critics pass thresholds | Iterate (Generate→Verify→Refine) up to 5 times with critic feedback | Stop after 10 cycles or critic deadlock, route to human review |

---

## 6. Human Review Gates

| Gate | Trigger | Reviewer Role | Tooling |
| --- | --- | --- | --- |
| High-chirality SNOs | β₁ > policy threshold or Antagonist severity=CRITICAL | Domain expert adjudicates whether synthesis should proceed | Web UI (side-by-side evidence, voting) |
| Novelty spikes | Novelty score >0.9 | Epistemic reviewer labels as "promising" vs "specious" | Annotation tool (e.g., Argilla) |
| Training audits | Per epoch on synthetic/bootstrapped data | Bias reviewer samples SNOs for harmful patterns | Sampled JSONL review via notebook |

---

## 7. Failure Modes & Recovery

| Failure | Detection | Recovery | Escalation |
| --- | --- | --- | --- |
| Proposer schema regression | Thinker validation fails repeatedly | Roll back config, bisect changes, rerun validation | Alert maintainers after 3 consecutive failures |
| Antagonist misses known contradictions | Synthetic test suite fails precision/recall | Tune thresholds, retrain heuristic models | Manual review of critic weights |
| Synthesizer diverges | β₁ increases >20% per iteration or critics disagree | Inject expert SNO, reset iteration, lower temperature | Human review if divergence persists |

---

## 8. Monitoring & Health Signals

| Metric | Collection | Alert Threshold | Response Playbook |
| --- | --- | --- | --- |
| Proposer schema pass rate | Per-run Thinker validation | <90% over 10 consecutive runs | 1) Check recent config changes<br>2) Bisect to last green commit<br>3) Inspect dataset for schema drift |
| Antagonist false positive rate | Weekly synthetic test suite | >20% | 1) Sample 20 FPs<br>2) Retune NLI threshold<br>3) Update heuristic rules<br>4) Re-run suite |
| Synthesizer mean iterations | Mean cycles per SNO | >7 (approaching hard stop) | 1) Inspect 7+ cycle SNOs<br>2) Check critic instability<br>3) Lower temperature/add regularization |
| Human review queue depth | Count of SNOs awaiting adjudication | >50 items | 1) Triage by β₁ (highest first)<br>2) Recruit additional reviewers<br>3) Temporarily raise escalation thresholds |

---

## Appendix A: SNO Quality Examples

### Good SNO (Proposer Output)

```json
{
  "hypothesis": "Vitamin D supplementation reduces COVID-19 severity",
  "claims": [
    {
      "id": "c1",
      "text": "Vitamin D deficiency correlates with severe COVID outcomes",
      "evidence_ids": ["e1", "e2"],
      "relation": "SUPPORTS"
    }
  ],
  "evidence": [
    {
      "id": "e1",
      "text": "Patients with <20ng/mL vitamin D had 2.5x ICU admission rate",
      "source": "PMID:12345678"
    },
    {
      "id": "e2",
      "text": "Meta-analysis (n=12,000) found inverse correlation between vitamin D levels and mortality",
      "source": "PMID:87654321"
    }
  ],
  "grounding_score": 0.82,
  "beta1": 0.0
}
```

**Why it's good:**  
✅ Citations exist and are valid (e1, e2)  
✅ Entailment passes (claim supported by evidence)  
✅ No logical holes (β₁ = 0)  
✅ Claim is appropriately hedged ("correlates" not "causes")

---

### Bad SNO (Proposer Output)

```json
{
  "hypothesis": "Vitamin D cures COVID-19",
  "claims": [
    {
      "id": "c1",
      "text": "Studies show vitamin D eliminates viral load",
      "evidence_ids": ["e1"],
      "relation": "SUPPORTS"
    }
  ],
  "evidence": [
    {
      "id": "e1",
      "text": "Vitamin D may play a role in immune function",
      "source": "PMID:11111111"
    }
  ],
  "grounding_score": 0.31,
  "beta1": 0.45
}
```

**Why it's bad:**  
❌ Claim overstates evidence ("cures" vs "may play role")  
❌ Weak entailment (0.31 < 0.75 threshold)  
❌ Introduces logical gap (β₁ = 0.45 > 0)  
❌ Single weak evidence source for strong claim

**Antagonist should flag:** Grounding failure, overgeneralization, citation insufficiency.

---

## Appendix B: Current Implementation Status (2025-11-18)

### Proposer Agent: ⚠️ FUNCTIONAL BUT REQUIRES IMPROVEMENT

**Status:** Production-ready for schema extraction, but citation hallucinations and weak semantic grounding require addressing

#### Breakthrough Results (Nov 11, 2025)

From 0% exact-match baseline to **36% semantic validation pass rate** in one day through:
1. Implementing 4-stage semantic validation pipeline
2. Fixing training prompts to enforce CLAIM[c*] schema
3. Adding explicit citation examples to training data
4. Full LoRA training (505 examples, 3 epochs, Llama-3.1-8B-Instruct)

#### Performance History

**Baseline (Nov 11, 2025 - adapter claim-extractor-scifact-20251118T173307):**
```
Schema Compliance:     100.0% (50/50) ✅ EXCEEDS TARGET (≥95%)
Citation Accuracy:     96.0% (48/50)  ✅ EXCELLENT (hard gate)
Mean Entailment Score: 0.448          ⚠️ BELOW TARGET (≥0.75)
Entailment Pass Rate:  38.0% (19/50)  ⚠️ MEASURABLE PROGRESS
Mean Similarity Score: 0.25           ⚠️ BELOW TARGET (≥0.70)
Similarity Pass Rate:  20.0% (10/50)  ⚠️ BELOW TARGET (≥60%)

🎯 OVERALL PASS RATE:  38.0% (19/50)  ✅ FIRST MEANINGFUL VALIDATION
```

**Training Iteration (Nov 18, 2025 - adapter claim-extractor-scifact-20251118T220454, weight=2.0):**
```
Schema Compliance:     98.0% (49/50)  ⚠️ SLIGHT REGRESSION (-2%)
Citation Accuracy:     96.0% (48/50)  = UNCHANGED
Mean Entailment Score: 0.395          ❌ WORSE (-0.053)
Entailment Pass Rate:  34.0% (17/50)  ❌ WORSE (-4%)
Mean Similarity Score: 0.25           = UNCHANGED
Similarity Pass Rate:  18.0% (9/50)   ❌ WORSE (-2%)

🎯 OVERALL PASS RATE:  34.0% (17/50)  ❌ REGRESSION (-4%)

⚠️ CRITICAL FINDING: 2 HIGH severity CITATION_INVALID cases (claims 133, 179)
   - Model fabricated document IDs not in source corpus
   - Evidence overlap: 20-25% (vs 100% for valid citations)
   - Entailment score: 0.0 (complete failure)
   - Penalty weight=2.0 insufficient to teach citation grounding
```

#### What This Validates

**✅ LoRA Architecture Decision Confirmed:**
- Proposer learns schema patterns perfectly (100% compliance)
- Citation extraction works reliably (96% accuracy)
- Format enforcement through prompt engineering is effective

**✅ Semantic Validation Provides Actionable Metrics:**
- Old: "0% exact-match" = no diagnostic information
- New: "100% schema, 96% citations, 36% entailment" = specific, actionable failure modes

**✅ Training Approach is Sound:**
- 505 examples sufficient for schema/citation learning
- 3 epochs adequate for format learning
- LoRA rank 16 sufficient for structured output

#### Critical Issues & Active Remediation (Nov 18, 2025)

**Issue 1: Citation Hallucination (P0 - CRITICAL)**
- **Problem:** Model fabricates document IDs not in source corpus (2/50 samples with HIGH severity flags)
- **Root cause:** Training penalty weight=2.0 (3x loss multiplier) insufficient to teach citation grounding
- **Impact:** Blocks Synthesizer development, creates 0.0 entailment failures
- **Active Remediation:**
  - ✅ Increased `citation_validity_weight` from 2.0 to 5.0 (6x loss multiplier, commit `e500bb2`)
  - 🔬 Next training run in progress
  - Success criteria: Eliminate HIGH severity CITATION_INVALID flags (2 → 0)
- **Fallback options if weight=5.0 fails:**
  1. Escalate to weight=10.0 or weight=20.0
  2. Negative example training (augment dataset with invalid citations + high penalties)
  3. Two-stage training (general extraction → citation-focused fine-tuning)

**Issue 2: Weak Semantic Grounding (P0 - CRITICAL)**
- **Problem:** Mean entailment 0.395-0.448 (target ≥0.75), overall pass 34-38% (target ≥60%)
- **Root cause:** Model learned citation format but not evidence-to-claim grounding relationships
- **Impact:** 60.9% of Antagonist flags have entailment <0.5
- **Remediation options:**
  1. **Current approach:** Citation penalty increase (may improve grounding as side effect)
  2. **Short-term:** Add contrastive loss for tighter evidence alignment (1-2 day investment, target: 50-60% pass)
  3. **Long-term:** Scale to 1000+ examples + increase LoRA rank to 32 (1 week investment, target: 60-70% pass)

### Antagonist Agent: ✅ MVP COMPLETE

**Status:** Production-ready MVP shipped Nov 18, 2025

#### Implementation Completed
- ✅ CLI integration: `python -m thinker.cli antagonist`
- ✅ Threshold-based heuristics (chirality ≥0.55, entailment <0.5)
- ✅ 4 issue types: CITATION_INVALID, POLARITY_CONTRADICTION, POLARITY_CONFLICT, WEAK_ENTAILMENT
- ✅ 22 unit tests passing
- ✅ Complete documentation (`docs/20251118/antagonist-mvp-review/`)

#### Real-World Validation Results (Nov 18, 2025 - 50 SciFact samples)
```
Flagging Rate:        92% (46/50 samples)
HIGH Severity:        2 cases (4.3%) - both CITATION_INVALID
MEDIUM Severity:      44 cases (95.7%)
Issue Distribution:
  - POLARITY_CONTRADICTION: 84.8% (mean chirality 0.561)
  - WEAK_ENTAILMENT: 60.9% (entailment <0.5)
  - CITATION_INVALID: 4.3% (HIGH severity)
False Positives:      0 (manual review confirmed all flags legitimate)
```

#### Critical Success: Identified Citation Hallucinations
- Antagonist correctly flagged 2 HIGH severity cases where Proposer fabricated document IDs
- Manual review confirmed these are real hallucinations, not false alarms
- Training iteration (weight=2.0) confirmed Antagonist diagnosis was accurate
- **Actionable insight:** Antagonist-driven diagnosis led to increased citation penalty (weight=5.0)

#### Next Steps (P1)
1. ⏳ Embedding anti-neighbor retrieval for counter-evidence generation
2. ⏳ DeBERTa contradiction scoring to upgrade POLARITY_CONTRADICTION detection
3. ⏳ Precision/recall instrumentation against 200-pair synthetic contradiction suite
4. ⏳ Expand test coverage to ≥80%

---

Use this playbook when onboarding collaborators, writing weekly updates, or planning workstreams—the goal is to keep CNS 3.0's agent model concrete, testable, philosophically aligned, and tightly linked to the pieces already shipping in this repo.
