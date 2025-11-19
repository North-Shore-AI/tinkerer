# Limited Evaluation Runs

Use the lightweight pipeline config `thinker/configs/pipeline_scifact_limited.yaml` when you want a
quick sanity check that only streams five SciFact samples and writes the completions/metrics to their
own files (`runs/thinker_eval/scifact_dev_eval_limited.jsonl`).

```bash
cd /home/home/p/g/North-Shore-AI/tinkerer
python -m thinker.cli eval \
  --config thinker/configs/pipeline_scifact_limited.yaml \
  --skip-validation
```

Because the config points at the same manifest as the full pipeline, make sure
`runs/latest_tinker_adapter.json` is up to date before running. All emitter artifacts for this
limited pass will land under `artifacts/eval/` with a unique run id, so the regular dashboards are
left untouched.

## Telemetry verification loop

The limited/“micro” configs are also the fastest way to populate the telemetry dashboard with
multi-point curves. The recommended flow mirrors `thinker.sh` menu entries:

1. `./thinker.sh` → option **4** (micro Tinker train)  
   - Emits per-step telemetry with timestamps, epoch/step counters, loss, citation invalid rate, and
     raw Tinker metrics under `artifacts/train/<run_id>/manifest.json`.
2. `./thinker.sh` → option **6** (micro eval)  
   - Logs per-sample semantic metrics (schema, citation, entailment, similarity, paraphrase) plus
     β₁, chirality, Fisher-Rao distance, and cumulative rate arrays into
     `artifacts/eval/<run_id>/manifest.json`.
3. `python -m thinker.cli antagonist` (uses the latest evaluation JSONL by default)  
   - Writes per-flag telemetry with timestamps, severity, trigger metrics, and issue types under
     `artifacts/antagonist/<run_id>/manifest.json`.
4. `python scripts/serve_dashboard.py --venv .venv`  
   - Open the dashboard to inspect the new runs: toggle per-step vs. cumulative loss, select
     evaluation metrics (entailment vs. chirality vs. citation validity), and review the antagonist
     severity chart + flag table.

Because each run is capped at ~5–10 samples, manifests stay small enough for local debugging but
still surface the same telemetry schema the full pipeline uses. The emitted manifests are
automatically indexed under `dashboard_data/index.json`, so you can stop/start the server without
losing historical run IDs.
