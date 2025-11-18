# Bregman Manifold Design Note for CNS 3.0

## 1. Purpose and Scope

The current CNS 3.0 workstream is still focused on building reliable Structured Narrative Objects (SNOs), but the long-term roadmaps in `cns3_gpt5.md` (§1, line 17) and `cns3_gemini_deepResearch.md` (§1, line 5) already lean on information-geometry language. This note captures how and when we should thread the **Bregman manifold** framing (à la Frank Nielsen) into the existing theory without blocking near-term Proposer deliverables. It is intentionally light on new proofs: the goal is to document the insertion points, dependencies, and checks that will let us formalize the dual-connection story once the basic SNO pipelines are green.

## 2. Existing Hooks We Can Reuse

1. **Hypothesis statistical manifold** – The main spec already states that hypothesis embeddings live on a Fisher-metric manifold obtained from exponential-family parameters (`cns3_gpt5.md`, line 54). This gives us the convex potential (log-partition) required to define a Hessian metric and, therefore, a Bregman divergence.
2. **Topological signals for critics** – The Gemini proposal formalizes SNOs as simplicial complexes and ties chirality to Betti numbers (`cns3_gemini_deepResearch.md`, lines 48–68). These β₁/β₀ invariants can serve as coordinates or regularizers inside a Bregman potential, ensuring the geometry stays compatible with the Logic critic.
3. **Agent responsibilities** – The CNS 3.0 Agent Playbook (Section 1) specifies that the Proposer, Antagonist, and Synthesizer exchange SNOs plus critic deltas. We can interpret those deltas as dual-coordinate residuals without changing the file-based handoff defined in Section 3 of the same doc.

## 3. What “Bregman Manifold” Adds

| Concept | Operational meaning | Mapping to existing artifacts |
| --- | --- | --- |
| Convex potential \(F\) | Strictly convex energy over hypothesis parameters \(\theta\) combining (a) log-normalizer of our exponential-family embedding and (b) penalties for β₁ holes. | Use the Fisher log-partition already implied in `cns3_gpt5.md` line 54, add β-weighted regularizer from Logic critic outputs stored in `artifacts/logic/*.jsonl`. |
| Hessian metric \(g = \nabla^2 F\) | Metric that matches Fisher locally but bakes in topology-aware curvature. | Extends Evidential Entanglement metrics (lines 15–23) with curvature terms computed from critic diagnostics. |
| Dual affine connections \(\nabla, \nabla^\*\) | Two torsion-free connections enabling “natural” vs. “expectation” coordinates for hypotheses. | Proposer operates in natural coordinates (\(\theta\)), Antagonist evaluates residuals in dual coordinates (\(\eta = \nabla F(\theta)\)). Synthesizer walks geodesics defined by averaging both views. |
| Bregman divergence \(D_F(\theta_1 \| \theta_2)\) | Asymmetric chirality metric that explains “who is overshooting” vs. “who is under-specifying.” | Replaces cosine distance inside the Chirality Score definition (Gemini doc §3.2) while remaining compatible with β₁-weighted conflict density. |

This framing is additive: we keep the Fisher-Rao guarantees already promised (`cns3_gemini_deepResearch.md`, lines 43–44) but gain a concrete recipe for dual-flat geometry that will resonate with information-geometry reviewers.

## 4. Suggested Updates to Theory Docs (Once SNO Basics Are Stable)

1. **`cns3_gpt5.md` §1.3 (Metrics & Geometry)** – Insert a paragraph after line 17 stating that the hypothesis manifold is realized as a Bregman/Hessian manifold induced by the exponential-family log-partition, referencing Nielsen’s tutorial (already cited via the pinned tweet) and clarifying that Fisher distances are recovered as the symmetric part of the divergence.
2. **`cns3_gemini_deepResearch.md` §3.2 (Relational Metrics)** – Replace the scalar “learned function approximating Fisher-Rao distance” with “Bregman divergence weighted by β₁.” Include the dual-coordinate interpretation: chirality flags are large when \(\theta\)-space proximity disagrees with \(\eta\)-space proximity.
3. **Append a short methodological note** (could be a new appendix) summarizing this design note so that reviewers understand we have a concrete landing pad even if the implementation happens post-MVP SNO validation.

## 5. Implementation Roadmap (Non-Blocking to Current Work)

### 5.1 Now (while Proposer SNOs are maturing)
- Log the natural parameters your current SNO embeddings already produce (e.g., logits or regression heads) so we can backfill \(\theta\).
- Ensure Logic critic outputs β₁ per SNO and persist them in the artifacts tree defined in the Agent Playbook §3.
- Add TODO comments or issue trackers referencing this note so we do not lose the insertion points after the current validation push.

### 5.2 Near Term (post-schema stability)
- Define the convex potential \(F(\theta) = A(\theta) + \lambda \cdot \beta_1(S)\), where \(A(\theta)\) is the log-partition from the exponential-family mapping and \(\lambda\) is tuned using Thinker validation metrics. This step is analytic and can be documented before any code changes.
- Update the chirality metric computation in the Synthesizer/Antagonist planning docs to reference \(D_F\). No runtime change is required until the critics emit the needed stats.

### 5.3 Medium Term (when Antagonist MVP lands)
- Teach the Antagonist to emit both \(\theta\)- and \(\eta\)-coordinate deltas per claim: \(\Delta_\theta = \theta_{prop} - \theta_{counter}\), \(\Delta_\eta = \eta_{prop} - \eta_{counter}\). Logging these is enough to test whether asymmetric divergences correlate with high-severity flags.
- Run an offline analysis on stored SNOs to see whether β₁ reduction correlates with \(D_F\) decrease; this becomes evidence for the Dialectical Convergence story.

### 5.4 Long Term (theory validation)
- Formalize the contraction-mapping proof using the Bregman projection framework—Bregman projections give a neat interpretation of the “evidence refresh” step in the Agent Playbook decision tree.
- Extend the Information Preservation theorem (lines 20 and 106 in `cns3_gpt5.md`) to show that Bregman-averaged synthesis preserves or increases observed Fisher information because \(D_F\) upper-bounds the drop.

## 6. Open Questions / Dependencies

1. **Choice of exponential family.** We loosely assume a natural-parameter mapping, but we still need to specify which sufficient statistics come from the Proposer’s encoder. A lightweight design review can settle this once the current SNO JSON schema performance stabilizes (target ≥95% per Agent Playbook §1.1).
2. **Critic access to gradients.** To compute \(\eta = \nabla F(\theta)\), critics need either analytic gradients or autodiff over the convex potential. This may require storing the log-partition form alongside the SNO manifest.
3. **Human interpretability.** Introducing asymmetric chirality metrics will change the way we report conflict severity to reviewers. We should coordinate with the Human Review Gate owners (Agent Playbook §6) to ensure the dashboards highlight the “direction” of disagreement, not just the magnitude.

## 7. Takeaway

No immediate engineering work is required: this note simply codifies how the Bregman manifold story will plug into CNS 3.0 once basic SNO generation and validation are reliable. When we are ready, we can cite this document alongside the existing Fisher-metric claims to show that the dual-geometry perspective is planned, consistent with the current theory, and traceable back to the Agent Playbook contracts.

