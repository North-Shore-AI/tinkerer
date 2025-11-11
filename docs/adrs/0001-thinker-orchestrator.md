# ADR 0001 · Adopt Thinker-Orchestrated Hugging Face + PEFT Workflow

**Status:** Accepted  
**Date:** 2025-11-10  
**Owners:** CNS Support Models team

---

## Context

- The brainstorm dossiers lay out a dual-track implementation plan: keep Tinker for managed LoRA runs while standing up a local Transformers + PEFT stack for fast iteration and GPU-free validation (`brainstorm/20251110/transformers-peft-tdd/01_IMPLEMENTATION_GUIDE.md`).  
- Code reviews and Claude feedback highlight a repeated failure mode: multi-hour Tinker jobs are being used to debug schema and data issues that should be caught locally (`brainstorm/20251110/003_tinker_tdd_claude.md`, `brainstorm/20251110/0010_claudeCode_feedback.md`).  
- The future “Thinker” framework is expected to become a first-class orchestrator that forces tests → dataset validation → local PEFT smoke run → remote Tinker run, so a canonical structure is required now.

## Decision

We will:

1. **Adopt “Thinker” as the orchestration layer** that loads a single YAML pipeline config defining tests, dataset checks, training backend, and evaluation targets.  
2. **Standardize on Hugging Face + PEFT for local smoke runs** (QLoRA-ready) as described in the implementation guide, including shared YAML hyperparameters.  
3. **Treat Thinker as the CLI that all contributors must use** for validation, training, and evaluation so the workflow is reproducible and backend-agnostic (local HF or remote Tinker).

## Rationale

- Provides a reproducible entry point for the end-to-end workflow the brainstorm docs describe, without rewriting every script.  
- Makes it trivial to toggle backends (HF PEFT for local smoke, Tinker for scaled runs) while keeping configs identical.  
- Encodes the validation-first philosophy so data/schema regressions cannot skip straight to GPU time.  
- Sets the foundation for later Elixir-based orchestration (“Thinker” ⇄ “Tinker”) without blocking current Python work.

## Consequences

- Contributors must express new experiments as Thinker pipeline configs; ad-hoc script execution is deprecated.  
- The Thinker package becomes a dependency of CNS Support Models; CI should run `thinker validate` + targeted tests.  
- Future features (e.g., additional critics, dataset templates) will be integrated via Thinker plugins/config rather than bespoke scripts.

## References

- `brainstorm/20251110/transformers-peft-tdd/01_IMPLEMENTATION_GUIDE.md`  
- `brainstorm/20251110/003_tinker_tdd_claude.md`  
- `brainstorm/20251110/0010_claudeCode_feedback.md`
