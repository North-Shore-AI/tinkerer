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
- **Next Research Tasks:**  
  1. Improve semantic grounding (contrastive loss, entailment critic-in-loop).  
  2. Broaden datasets (FEVER config/tests, temporal corpora).  
  3. Auto-generate SNO manifests (graph export + metadata hashes).

---

### 1.2 Antagonist Agent

- **Inputs:** Proposer SNOs, critic thresholds (target β₁, chirality score).  
- **Outputs:** Counter-SNOs highlighting logical loops, unsupported claims, bias signatures, plus quantified β₁/chirality deltas (diagnostic, not rewarded).  
- **Planned Mechanics:**  
  - LLM prompting to "mirror" or negate hypotheses (per `cns3/20251109_revised_cns_proposal_for_thinking_machines.md`).  
  - Retrieval of conflicting evidence; scoring via Logic + Bias critics.  
- **Implementation TODOs:**  
  1. Stand up heuristic passes (regex negation detection, embedding anti-neighbors, lightweight NLI contradiction filter).  
  2. Record flagged issues in structured JSON (claim_id, issue_type, critic_score).  
  3. Feed outputs into Thinker as optional pre-synthesis "stress tests."

#### Antagonist Success Metrics

- **Precision ≥0.8** on a synthetic contradiction test suite (no false alarms).
  - **Measurement:** Weekly run against 200 hand-labeled contradiction pairs (100 true contradictions, 100 spurious). Precision = TP/(TP+FP).
- **Recall ≥0.7** on a known-contradiction validation set (doesn't miss real issues).
  - **Measurement:** Same 200-pair suite. Recall = TP/(TP+FN).
- **β₁ quantification accuracy** within ±10% of graph-theory ground truth on labeled samples.
  - **Measurement:** Compare Antagonist's β₁ estimate to ground-truth Betti numbers on 50 manually constructed SNO graphs (validated by topologists).
- **Actionable flag rate:** ≥80% of HIGH-severity flags lead to Proposer refinement or human escalation.
  - **Measurement:** Track disposition of HIGH flags over 30 days: (refined + escalated)/total_flags ≥ 0.8.

**Anti-pattern:** Rewarding "more issues found."  
**Desired pattern:** Rewarding accurate issue detection that drives resolution.

---

### 1.3 Synthesizer Agent

- **Inputs:** High-chirality/high-entanglement SNO pairs, Antagonist deltas, critic weights.  
- **Outputs:** Candidate synthesized SNOs (hypothesis, reasoning graph, evidence set, trust score), manifest entries for downstream evaluation.  
- **Planned Stack:**  
  - Base models: Llama‑3.1‑70B (development) → Qwen3‑235B MoE (production).  
  - Constrained decoding (KCTS + citation enforcement).  
  - Critic-guided refinement loop (Generate → Verify → Refine).  
- **Pre-work:**  
  1. Finalize critic interfaces (Grounding, Logic, Novelty) as callable services.  
  2. Define SNO manifest schema (superset of `runs/latest_tinker_adapter.json`).  
  3. Prototype with Thinker eval harness by swapping Tinker sampling backend once adapter exists.

---

## 2. Supporting Operational Agents

These agents ensure the core pipeline is reproducible and review-ready even before the full dialectical loop is live.

---

### 2.1 Data & Validation Agent

- **Role:** Enforce ADR‑0002 "test-before-GPU" gate.  
- **Mechanics:** Thinker CLI `validate` stage running CNS pytest suite + JSONL validator (`thinker/validation.py`).  
- **Runbook:** `thinker.sh` recommends **5 (data setup) → 1 (validate)** prior to any training/eval.  
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
- **Next Steps:** Plug in critic scores (Grounding/Logic/Novelty) once available, add exact-match vs. paraphrase dashboards, expose semantic similarity metrics to avoid over-reliance on strict string match.

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

Use this playbook when onboarding collaborators, writing weekly updates, or planning workstreams—the goal is to keep CNS 3.0's agent model concrete, testable, philosophically aligned, and tightly linked to the pieces already shipping in this repo.
