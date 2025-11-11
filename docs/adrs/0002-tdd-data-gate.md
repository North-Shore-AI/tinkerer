# ADR 0002 · Enforce TDD Data Validation Gate Before Any GPU Training

**Status:** Accepted  
**Date:** 2025-11-10  
**Owners:** CNS Support Models team

---

## Context

- Multiple retrospectives document that SciFact LoRA runs consumed hours of Tinker compute only to reveal 8.8 % semantic match and 0 % relation accuracy because converters emitted paraphrased or malformed evidence (`brainstorm/20251110/0010_claudeCode_feedback.md`, `brainstorm/20251110/002_tinker_claude.md`).  
- The dedicated TDD memo prescribes a specific pytest layout, golden fixtures, Hypothesis property tests, and dataset validators to gate every run before touching a GPU (`brainstorm/20251110/003_tinker_tdd_claude.md`).  
- The Transformers + PEFT implementation guide assumes those tests already exist and explicitly instructs reusing them verbatim (`brainstorm/20251110/transformers-peft-tdd/01_IMPLEMENTATION_GUIDE.md`).

## Decision

1. **All CNS pipelines MUST run the full TDD suite (pytest + dataset validator) before initiating either local PEFT or Tinker training.**  
2. **Thinker’s `validate` command is the enforcement point**: it calls pytest with the required markers and runs schema-aware JSONL validation; `thinker train`/`thinker run` will auto-trigger validation unless explicitly skipped (only for debugging).  
3. **Conversion scripts, schema modules, and dataset outputs cannot be merged without new/updated tests** that catch the regression which motivated the change.

## Rationale

- Dramatically shortens the feedback loop: schema bugs surface in seconds instead of after 2–3 h GPU jobs.  
- Aligns with Claude’s recommendation to break the critics↔SNO circular dependency by validating core hypotheses (CLAIM[c1] literal copy, evidence verbatim, deterministic conversions) before further automation.  
- Prevents costly regressions as we port the same data into both local PEFT and Tinker backends.

## Consequences

- Contributors must keep fixtures/golden examples in sync with conversion logic; PRs touching schemas will include test updates.  
- CI must fail fast on `thinker validate`; skipping validation is grounds for blocking a merge.  
- Training/eval scripts can assume validated inputs, simplifying runtime checks but requiring higher discipline in tests.

## References

- `brainstorm/20251110/003_tinker_tdd_claude.md`  
- `brainstorm/20251110/0010_claudeCode_feedback.md`  
- `brainstorm/20251110/002_tinker_claude.md`  
- `brainstorm/20251110/transformers-peft-tdd/01_IMPLEMENTATION_GUIDE.md`
