# ADR 0003 · Maintain Dual Training Backends (Tinker + DIY Hugging Face)

**Status:** Accepted  
**Date:** 2025-11-10  
**Owners:** CNS Support Models team

---

## Context

- The brainstorm notes compare staying on Tinker vs. self-hosting LoRA, noting that small SciFact runs already complete within preview quotas but that custom schedulers, gradient logging, and >1 k-step sweeps will require full control (`brainstorm/20251110/0012_tinker_vs_diy.md`, `brainstorm/20251110/0013_tinker_diy_thoughts.md`).  
- The implementation guide explicitly describes a Hugging Face + PEFT QLoRA stack that mirrors the Tinker configs so we can test locally and fall back when Tinker API changes or access is revoked (`brainstorm/20251110/transformers-peft-tdd/01_IMPLEMENTATION_GUIDE.md`).  
- Claude’s feedback insists on decoupling from Tinker to avoid being blocked by a private beta and to enable reproducible research artifacts (`brainstorm/20251110/0010_claudeCode_feedback.md`).

## Decision

1. **Keep Tinker as the default training backend for managed LoRA runs** (fast iteration, hosted GPUs, provenance logging).  
2. **Implement and maintain a functionally equivalent Hugging Face + PEFT training path** (QLoRA-capable) that shares the same configs/metrics for local smoke tests, offline work, and future migration.  
3. **Expose backend selection through Thinker configs/CLI** (`hf_peft`, `tinker`, future additions) so every experiment can declare the required environment but remain reproducible across backends.

## Rationale

- Protects us from platform risk (pricing, quotas, API drift) while still exploiting Tinker’s strengths for quick remote jobs.  
- Enables deterministic local debugging (e.g., controlled overfit tests, gradient inspection) that Tinker does not surface.  
- Keeps adapter hyperparameters + dataset preparation identical regardless of backend, eliminating divergence between “local” and “cloud” paths.

## Consequences

- We must document backend-specific requirements (GPU memory, HF tokens, Tinker credentials) and keep both paths tested.  
- Thinker validation + training steps must detect backend mismatches early (e.g., prompt for HF auth if `hf_peft` selected).  
- Release artifacts (checkpoints, eval logs) should note the backend used so regressions can be reproduced.

## References

- `brainstorm/20251110/0012_tinker_vs_diy.md`  
- `brainstorm/20251110/0013_tinker_diy_thoughts.md`  
- `brainstorm/20251110/transformers-peft-tdd/01_IMPLEMENTATION_GUIDE.md`  
- `brainstorm/20251110/0010_claudeCode_feedback.md`
