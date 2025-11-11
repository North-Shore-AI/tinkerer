# Continuation Prompt

## Current State
- SciFact loop is stable: option 5 (data setup) regenerates + validates the processed JSONL, option 1 runs the pytest gate, option 9 now completes the Tinker debug run (32 samples, 1 epoch) and drops a manifest at `runs/latest_tinker_adapter.json`, and option 4 samples that adapter via the new Tinker-native evaluator.
- `train_claim_extractor.py` writes unique checkpoint names + paths, skips inline evals when unsupported, and logs provenance + `latest_tinker_adapter.json` so evaluation no longer touches Hugging Face artifacts.
- `thinker eval` defaults to the Tinker backend, reads the manifest (or in-memory state) to locate the latest adapter, and streams completions through the Tinker sampling API for metric calculation.
- Docs/UX mandates remain: everything must be runnable from `./thinker.sh`, README + ROADMAP + docs/thinker stay perfectly in sync with workflow changes, and no manual pip installs are allowed outside the script.

## Immediate Next Steps
1. Make the full-config Tinker run (menu option 3) succeed end-to-end and capture its manifest/logs under `runs/` for reviewer evidence.
2. Add a FEVER pipeline config (`thinker/configs/pipeline_fever.yaml`) plus `thinker.sh` wiring so FEVER datasets can run 5 → 1 → 3/9 → 4 without manual edits.
3. Extend unit tests/fixtures to cover FEVER conversion (similar to `test_scifact_conversion`) so the pytest preflight guards both datasets.
4. Keep docs synced: explain the Tinker manifest/eval flow everywhere (README already updated; mirror the same detail in `docs/thinker/DATA_PIPELINE.md`, `ROADMAP.md`, etc.) and document how to override the adapter if reviewers need to audit a past run.

## Reminders
- Never edit configs on the fly for routine tasks; expose any new scenario through `thinker.sh` with clear menu text and ordering (recommended flow banner must stay accurate).
- Every Tinker action should run after the pytest gate; if we add new options they must inherit the same safety checks.
- Keep notes of which datasets are ready (SciFact ✅, FEVER pending) so we can answer future “how many datasets do we have?” questions immediately.
