"""Ultra-simple HTTP server that surfaces dashboard_data/index.json."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import parse_qs, urlparse

from . import DEFAULT_PORT

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_ROOT = REPO_ROOT / "dashboard_data"
INDEX_PATH = DASHBOARD_ROOT / "index.json"
TRAIN_METRICS = ["final_loss", "citation_invalid_rate"]
EVAL_METRICS = [
    "schema_compliance_rate",
    "citation_accuracy_rate",
    "mean_entailment_score",
    "std_entailment_score",
    "mean_semantic_similarity",
    "std_semantic_similarity",
    "overall_pass_rate",
]
ANTAGONIST_METRICS = ["flag_rate", "flagged_records", "high_flags", "medium_flags"]


def _load_index() -> Dict[str, Any]:
    if not INDEX_PATH.exists():
        return {"runs": []}
    with INDEX_PATH.open("r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError:
            return {"runs": []}


def _load_manifest(run_id: str) -> Dict[str, Any] | None:
    for entry in _iter_manifests(include_entry=True):
        manifest, index_entry = entry
        if index_entry.get("run_id") == run_id:
            return manifest
    return None


def _iter_manifests(*, include_entry: bool = False) -> Iterable[Any]:
    index = _load_index()
    for entry in index.get("runs", []):
        manifest_path = entry.get("manifest_path")
        if not manifest_path:
            continue
        path = REPO_ROOT / manifest_path
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        except json.JSONDecodeError:
            continue
        if include_entry:
            yield manifest, entry
        else:
            yield manifest


def _render_runs_table(runs: List[Dict[str, Any]]) -> str:
    if not runs:
        return "<p>No runs found yet. Execute thinker CLI commands to emit metrics.</p>"
    rows = []
    for entry in runs:
        run_id = entry.get("run_id", "-")
        stage = entry.get("stage", "-")
        ts = entry.get("timestamp", "-")
        rows.append(
            f"<tr><td>{stage}</td><td>{run_id}</td><td>{ts}</td>"
            f"<td><a href=\"/manifest?run_id={run_id}\">view</a></td></tr>"
        )
    return """
<table>
  <thead>
    <tr><th>Stage</th><th>Run ID</th><th>Timestamp (UTC)</th><th>Manifest</th></tr>
  </thead>
  <tbody>
{rows}
  </tbody>
</table>
""".replace("{rows}", "\n".join(rows))


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler signature)
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index", "/index.html"):
            self._handle_index()
        elif parsed.path == "/manifest":
            params = parse_qs(parsed.query)
            run_id = params.get("run_id", [None])[0]
            if not run_id:
                self._write_json({"error": "run_id required"}, status=400)
                return
            manifest = _load_manifest(run_id)
            if manifest is None:
                write_json(self, {"error": f"manifest for run {run_id} not found"}, status=404)
                return
            write_json(self, manifest)
        elif parsed.path == "/api/metrics":
            payload = _collect_metrics_payload()
            write_json(self, payload)
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def _handle_index(self) -> None:
        index = _load_index()
        runs = index.get("runs", [])
        html_template = """
<!DOCTYPE html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <title>Thinker Metrics Dashboard</title>
    <style>
      body { font-family: sans-serif; margin: 2rem; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #ccc; padding: 0.5rem; text-align: left; }
      th { background: #f4f4f4; }
      .chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem; margin-top: 2rem; }
      .chart-card { border: 1px solid #ddd; border-radius: 8px; padding: 1rem; min-height: 320px; position: relative; }
      canvas { width: 100%; height: 260px !important; max-height: 260px; }
      .chart-empty { color: #666; font-style: italic; }
      .metrics-json { background: #111; color: #0f0; padding: 1rem; border-radius: 6px; overflow-x: auto; font-size: 0.85rem; }
      .chart-controls { display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 0.75rem; }
      .chart-controls label { font-weight: 600; }
      .metric-toggle-group label { font-weight: normal; border: 1px solid #ddd; border-radius: 4px; padding: 0.15rem 0.5rem; background: #f9f9f9; }
      .chart-summary { font-size: 0.9rem; color: #444; margin-bottom: 0.5rem; }
      .flag-table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; }
      .flag-table th { background: #fafafa; }
      details summary { cursor: pointer; font-weight: 600; margin-bottom: 0.5rem; }
    </style>
  </head>
  <body>
    <h1>Thinker Metrics Dashboard</h1>
    <p>Serving data from __INDEX_PATH__</p>
    __RUNS_TABLE__
    <section>
      <h2>Metric Trends</h2>
      <div class=\"chart-grid\">
        <div class=\"chart-card\">
          <h3>Training Metrics</h3>
          <canvas id=\"trainChart\" height=\"260\"></canvas>
          <p class=\"chart-empty\" id=\"trainEmpty\">No training runs yet.</p>
        </div>
        <div class=\"chart-card\">
          <h3>Evaluation (Key Rates)</h3>
          <canvas id=\"evalChart\" height=\"260\"></canvas>
          <p class=\"chart-empty\" id=\"evalEmpty\">No evaluation runs yet.</p>
        </div>
        <div class=\"chart-card\">
          <h3>Antagonist Flags</h3>
          <canvas id=\"antChart\" height=\"260\"></canvas>
          <p class=\"chart-empty\" id=\"antEmpty\">No Antagonist runs yet.</p>
        </div>
      </div>
    </section>
    <section>
      <h2>Training Run Detail</h2>
      <div class=\"chart-controls\">
        <label for=\"trainRunSelect\">Run:</label>
        <select id=\"trainRunSelect\"></select>
        <label for=\"trainDetailMode\">View:</label>
        <select id=\"trainDetailMode\">
          <option value=\"per-step\" selected>Per-step</option>
          <option value=\"cumulative\">Cumulative average</option>
        </select>
      </div>
      <div class=\"chart-card\">
        <canvas id=\"trainDetailChart\" height=\"260\"></canvas>
        <p class=\"chart-empty\" id=\"trainDetailEmpty\">Select a training run to view per-step metrics.</p>
      </div>
    </section>
    <section>
      <h2>Evaluation Run Detail</h2>
      <div class=\"chart-controls\" style=\"align-items:flex-start;\">
        <div>
          <label for=\"evalRunSelect\">Run:</label>
          <select id=\"evalRunSelect\"></select>
        </div>
        <div id=\"evalMetricControls\" class=\"chart-controls metric-toggle-group\"></div>
      </div>
      <div class=\"chart-card\">
        <canvas id=\"evalDetailChart\" height=\"260\"></canvas>
        <p class=\"chart-empty\" id=\"evalDetailEmpty\">Select a run and metrics to view per-sample telemetry.</p>
      </div>
    </section>
    <section>
      <h2>Antagonist Detail</h2>
      <div class=\"chart-controls\">
        <label for=\"antRunSelect\">Run:</label>
        <select id=\"antRunSelect\"></select>
      </div>
      <div class=\"chart-grid\">
        <div class=\"chart-card\">
          <div id=\"antRunSummary\" class=\"chart-summary\"></div>
          <canvas id=\"antDetailChart\" height=\"260\"></canvas>
          <p class=\"chart-empty\" id=\"antDetailEmpty\">Select a run to view flag distribution.</p>
        </div>
        <div class=\"chart-card\">
          <h3>Flagged Claims</h3>
          <div id=\"antFlagEmpty\" class=\"chart-empty\">Select a run to view flags.</div>
          <table id=\"antFlagTable\" class=\"flag-table\">
            <thead>
              <tr><th>Claim</th><th>Severity</th><th>Chirality</th><th>Entailment</th><th>Citation</th><th>Issues</th></tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </section>
    <section>
      <h2>Raw Metrics Snapshot</h2>
      <details open>
        <summary>Toggle raw payload</summary>
        <pre id=\"metricsDump\" class=\"metrics-json\">Loading metrics…</pre>
      </details>
    </section>
    <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
    <script>
      const formatPercent = (value) => {
        if (value === null || value === undefined || Number.isNaN(value)) {
          return 'n/a';
        }
        return (Number(value) * 100).toFixed(1) + '%';
      };

      const evalMetricDefs = [
        { key: 'entailment_score', label: 'Entailment', color: '#d62728', axis: 'y', defaultChecked: true },
        { key: 'semantic_similarity', label: 'Semantic Similarity', color: '#9467bd', axis: 'y', defaultChecked: true },
        { key: 'citation_valid', label: 'Citation Valid', color: '#2ca02c', axis: 'y1', defaultChecked: true, asPercent: true },
        { key: 'overall_pass', label: 'Overall Pass', color: '#17becf', axis: 'y1', defaultChecked: false, asPercent: true },
        { key: 'chirality_score', label: 'Chirality', color: '#bcbd22', axis: 'y', defaultChecked: false },
        { key: 'beta1', label: 'β₁', color: '#8c564b', axis: 'y1', defaultChecked: false },
      ];

      let trainDetailChart = null;
      let trainDetailData = [];
      let trainDetailMode = 'per-step';
      let evalDetailChart = null;
      let evalDetailSamples = [];
      let selectedEvalMetrics = new Set(evalMetricDefs.filter((def) => def.defaultChecked !== false).map((def) => def.key));
      let antDetailChart = null;

      async function initCharts() {
        try {
          const response = await fetch('/api/metrics');
          const data = await response.json();
          console.log('[dashboard] metrics payload', data);
          renderLineChart('trainChart', 'trainEmpty', data.train, [
            { key: 'final_loss', label: 'Final Loss', color: '#ff7f50' },
            { key: 'citation_invalid_rate', label: 'Citation Invalid Rate', color: '#1f77b4', yAxisID: 'y1', tooltipFormatter: formatPercent },
          ], { y: { beginAtZero: true }, y1: { suggestedMin: 0, suggestedMax: 1 } });

          renderLineChart('evalChart', 'evalEmpty', data.evaluation, [
            { key: 'schema_compliance_rate', label: 'Schema', color: '#1f77b4', tooltipFormatter: formatPercent },
            { key: 'citation_accuracy_rate', label: 'Citation', color: '#2ca02c', tooltipFormatter: formatPercent },
            { key: 'mean_entailment_score', label: 'Mean Entailment', color: '#d62728', tooltipFormatter: formatPercent },
            { key: 'std_entailment_score', label: 'Entailment σ', color: '#d62728', borderDash: [4, 4], tooltipFormatter: formatPercent },
            { key: 'mean_semantic_similarity', label: 'Mean Similarity', color: '#9467bd', tooltipFormatter: formatPercent },
            { key: 'std_semantic_similarity', label: 'Similarity σ', color: '#9467bd', borderDash: [4, 4], tooltipFormatter: formatPercent },
            { key: 'overall_pass_rate', label: 'Overall Pass', color: '#8c564b', tooltipFormatter: formatPercent },
          ], { y: { suggestedMin: 0, suggestedMax: 1 } });

          renderLineChart('antChart', 'antEmpty', data.antagonist, [
            { key: 'flag_rate', label: 'Flag Rate', color: '#17becf', tooltipFormatter: formatPercent },
            { key: 'flagged_records', label: 'Flag Count', color: '#bcbd22', yAxisID: 'y1' },
            { key: 'high_flags', label: 'High Severity', color: '#ff7f0e', yAxisID: 'y1' },
            { key: 'medium_flags', label: 'Medium Severity', color: '#9467bd', yAxisID: 'y1' },
          ], { y: { suggestedMin: 0, suggestedMax: 1 }, y1: { beginAtZero: true } });

          setupTrainingRunDetail(data.train);
          setupEvalRunDetail(data.evaluation);
          setupAntagonistDetail(data.antagonist);

          const dump = document.getElementById('metricsDump');
          if (dump) {
            dump.textContent = JSON.stringify(data, null, 2);
          }
        } catch (error) {
          console.error('Failed to load metrics', error);
          const dump = document.getElementById('metricsDump');
          if (dump) {
            dump.textContent = 'Failed to load metrics: ' + error;
          }
        }
      }

      function renderLineChart(canvasId, emptyId, dataset, seriesDefs, options) {
        const canvas = document.getElementById(canvasId);
        const empty = document.getElementById(emptyId);
        if (!canvas || !empty) {
          return;
        }
        if (!dataset || dataset.labels.length === 0) {
          canvas.style.display = 'none';
          empty.style.display = 'block';
          return;
        }
        canvas.style.display = 'block';
        empty.style.display = 'none';

        const labels = dataset.labels.map((ts) => new Date(ts).toLocaleString());
        const seriesData = seriesDefs.map((def) => {
          const rawValues = dataset.series[def.key] || [];
          const values = rawValues.map((val) => (val === null || val === undefined ? null : Number(val)));
          const ds = {
            label: def.label,
            data: values,
            borderColor: def.color,
            backgroundColor: def.color,
            tension: 0.25,
            fill: false,
            yAxisID: def.yAxisID || 'y',
            spanGaps: true,
          };
          ds.key = def.key;
          if (def.borderDash) {
            ds.borderDash = def.borderDash;
          }
          if (def.tooltipFormatter) {
            ds._formatter = def.tooltipFormatter;
          }
          return ds;
        });

        const baseScales = {
          y: Object.assign({ beginAtZero: false, title: { display: true, text: 'Value' } }, (options && options.y) || {}),
        };
        if (options && options.y1) {
          baseScales.y1 = Object.assign({ position: 'right', grid: { drawOnChartArea: false } }, options.y1);
        }

        new Chart(canvas.getContext('2d'), {
          type: 'line',
          data: { labels, datasets: seriesData },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            stacked: false,
            scales: baseScales,
            plugins: {
              legend: { position: 'bottom' },
              tooltip: {
                callbacks: {
                  label: function(context) {
                    const datasetDef = seriesData[context.datasetIndex];
                    const formatter = datasetDef._formatter || (options && options.tooltipFormatter);
                    const value = context.parsed.y;
                    if (value === null || value === undefined) {
                      return context.dataset.label + ': n/a';
                    }
                    if (formatter) {
                      return context.dataset.label + ': ' + formatter(value, datasetDef.key);
                    }
                    return context.dataset.label + ': ' + value;
                  },
                },
              },
            },
          },
        });
      }

      function setupTrainingRunDetail(trainData) {
        const select = document.getElementById('trainRunSelect');
        const modeSelect = document.getElementById('trainDetailMode');
        if (!select) {
          return;
        }
        select.innerHTML = '';
        if (!trainData || !trainData.run_ids || trainData.run_ids.length === 0) {
          const opt = document.createElement('option');
          opt.value = '';
          opt.textContent = 'No training runs yet';
          select.appendChild(opt);
          select.disabled = true;
          if (modeSelect) {
            modeSelect.disabled = true;
          }
          renderTrainingDetailChart([]);
          return;
        }
        select.disabled = false;
        if (modeSelect) {
          modeSelect.disabled = false;
          modeSelect.value = trainDetailMode;
          modeSelect.onchange = (event) => {
            trainDetailMode = event.target.value;
            renderTrainingDetailChart(trainDetailData, trainDetailMode);
          };
        }
        trainData.run_ids.forEach((runId, idx) => {
          const ts = trainData.labels && trainData.labels[idx] ? new Date(trainData.labels[idx]).toLocaleString() : runId;
          const opt = document.createElement('option');
          opt.value = runId;
          opt.textContent = ts + ' (' + runId + ')';
          select.appendChild(opt);
        });
        select.onchange = (event) => {
          loadTrainingRunDetail(event.target.value);
        };
        if (select.value) {
          loadTrainingRunDetail(select.value);
        } else {
          renderTrainingDetailChart([]);
        }
      }

      async function loadTrainingRunDetail(runId) {
        if (!runId) {
          trainDetailData = [];
          renderTrainingDetailChart([]);
          return;
        }
        try {
          const response = await fetch('/manifest?run_id=' + encodeURIComponent(runId));
          if (!response.ok) {
            throw new Error('manifest lookup failed (' + response.status + ')');
          }
          const manifest = await response.json();
          const payload = manifest.payload || {};
          const metrics = payload.metrics || {};
          const stepMetrics = metrics.step_metrics || (metrics.training_report && metrics.training_report.step_metrics) || [];
          trainDetailData = stepMetrics;
          renderTrainingDetailChart(stepMetrics, trainDetailMode);
        } catch (error) {
          console.error('Failed to load training run detail', error);
          trainDetailData = [];
          renderTrainingDetailChart([]);
        }
      }

      function renderTrainingDetailChart(stepMetrics, mode = 'per-step') {
        const canvas = document.getElementById('trainDetailChart');
        const empty = document.getElementById('trainDetailEmpty');
        if (!canvas || !empty) {
          return;
        }
        if (!stepMetrics || stepMetrics.length === 0) {
          canvas.style.display = 'none';
          empty.style.display = 'block';
          if (trainDetailChart) {
            trainDetailChart.destroy();
            trainDetailChart = null;
          }
          return;
        }
        canvas.style.display = 'block';
        empty.style.display = 'none';
        const labels = stepMetrics.map((entry, idx) => {
          if (entry.timestamp) {
            return new Date(entry.timestamp).toLocaleString();
          }
          if (entry.step !== undefined) {
            return 'Step ' + entry.step;
          }
          return 'Point ' + (idx + 1);
        });
        const lossValues = stepMetrics.map((entry) => (entry.loss === undefined ? null : Number(entry.loss)));
        const citationValues = stepMetrics.map((entry) => (entry.citation_invalid_rate === undefined ? null : Number(entry.citation_invalid_rate)));

        const lossSeries = mode === 'cumulative' ? runningAverage(lossValues) : lossValues;
        const citationSeries = mode === 'cumulative' ? runningAverage(citationValues) : citationValues;

        const datasets = [
          {
            label: mode === 'cumulative' ? 'Loss (running avg)' : 'Loss',
            data: lossSeries,
            borderColor: '#ff7f50',
            backgroundColor: '#ff7f50',
            tension: 0.25,
            fill: false,
            yAxisID: 'y',
            spanGaps: true,
          },
          {
            label: mode === 'cumulative' ? 'Citation Invalid Rate (avg)' : 'Citation Invalid Rate',
            data: citationSeries,
            borderColor: '#1f77b4',
            backgroundColor: '#1f77b4',
            tension: 0.25,
            fill: false,
            yAxisID: 'y1',
            spanGaps: true,
          },
        ];

        const options = {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: 'index', intersect: false },
          stacked: false,
          scales: {
            y: { beginAtZero: false, title: { display: true, text: 'Loss' } },
            y1: { position: 'right', beginAtZero: true, suggestedMax: 1, grid: { drawOnChartArea: false }, title: { display: true, text: 'Citation Rate' } },
          },
          plugins: {
            legend: { position: 'bottom' },
            tooltip: {
              callbacks: {
                label: function(context) {
                  const value = context.parsed.y;
                  if (context.dataset.yAxisID === 'y1' && value !== null && value !== undefined) {
                    return context.dataset.label + ': ' + (value * 100).toFixed(2) + '%';
                  }
                  return context.dataset.label + ': ' + value;
                },
              },
            },
          },
        };

        if (trainDetailChart) {
          trainDetailChart.destroy();
        }
        trainDetailChart = new Chart(canvas.getContext('2d'), {
          type: 'line',
          data: { labels, datasets },
          options,
        });
      }

      function runningAverage(values) {
        const result = [];
        let total = 0;
        let count = 0;
        values.forEach((value) => {
          if (value === null || value === undefined || Number.isNaN(value)) {
            result.push(count ? total / count : null);
            return;
          }
          total += Number(value);
          count += 1;
          result.push(total / count);
        });
        return result;
      }

      function setupEvalRunDetail(evalData) {
        const select = document.getElementById('evalRunSelect');
        const controls = document.getElementById('evalMetricControls');
        if (!select || !controls) {
          return;
        }
        renderEvalMetricControls();
        select.innerHTML = '';
        if (!evalData || !evalData.run_ids || evalData.run_ids.length === 0) {
          const opt = document.createElement('option');
          opt.value = '';
          opt.textContent = 'No evaluation runs yet';
          select.appendChild(opt);
          select.disabled = true;
          renderEvalDetailChart([]);
          controls.style.display = 'none';
          return;
        }
        controls.style.display = 'flex';
        select.disabled = false;
        evalData.run_ids.forEach((runId, idx) => {
          const ts = evalData.labels && evalData.labels[idx] ? new Date(evalData.labels[idx]).toLocaleString() : runId;
          const opt = document.createElement('option');
          opt.value = runId;
          opt.textContent = ts + ' (' + runId + ')';
          select.appendChild(opt);
        });
        select.onchange = (event) => {
          loadEvalRunDetail(event.target.value);
        };
        if (select.value) {
          loadEvalRunDetail(select.value);
        } else {
          renderEvalDetailChart([]);
        }
      }

      function renderEvalMetricControls() {
        const controls = document.getElementById('evalMetricControls');
        if (!controls) {
          return;
        }
        controls.innerHTML = '';
        evalMetricDefs.forEach((def) => {
          const label = document.createElement('label');
          const checkbox = document.createElement('input');
          checkbox.type = 'checkbox';
          checkbox.value = def.key;
          checkbox.checked = selectedEvalMetrics.has(def.key);
          checkbox.onchange = (event) => {
            if (event.target.checked) {
              selectedEvalMetrics.add(def.key);
            } else {
              selectedEvalMetrics.delete(def.key);
            }
            renderEvalDetailChart(evalDetailSamples);
          };
          label.appendChild(checkbox);
          label.appendChild(document.createTextNode(' ' + def.label));
          controls.appendChild(label);
        });
      }

      async function loadEvalRunDetail(runId) {
        if (!runId) {
          evalDetailSamples = [];
          renderEvalDetailChart([]);
          return;
        }
        try {
          const response = await fetch('/manifest?run_id=' + encodeURIComponent(runId));
          if (!response.ok) {
            throw new Error('manifest lookup failed (' + response.status + ')');
          }
          const manifest = await response.json();
          const payload = manifest.payload || {};
          const metrics = payload.metrics || {};
          evalDetailSamples = metrics.per_sample_metrics || [];
          renderEvalDetailChart(evalDetailSamples);
        } catch (error) {
          console.error('Failed to load evaluation run detail', error);
          evalDetailSamples = [];
          renderEvalDetailChart([]);
        }
      }

      function renderEvalDetailChart(samples) {
        const canvas = document.getElementById('evalDetailChart');
        const empty = document.getElementById('evalDetailEmpty');
        if (!canvas || !empty) {
          return;
        }
        if (!samples || samples.length === 0 || selectedEvalMetrics.size === 0) {
          canvas.style.display = 'none';
          empty.style.display = 'block';
          if (evalDetailChart) {
            evalDetailChart.destroy();
            evalDetailChart = null;
          }
          return;
        }
        canvas.style.display = 'block';
        empty.style.display = 'none';
        const labels = samples.map((sample, idx) => {
          if (sample.timestamp) {
            return new Date(sample.timestamp).toLocaleString();
          }
          if (sample.index !== undefined) {
            return 'Sample ' + sample.index;
          }
          return 'Sample ' + (idx + 1);
        });

        const datasets = [];
        selectedEvalMetrics.forEach((key) => {
          const def = evalMetricDefs.find((entry) => entry.key === key);
          if (!def) {
            return;
          }
          const values = samples.map((sample) => {
            const raw = sample[key];
            if (raw === null || raw === undefined) {
              return null;
            }
            if (typeof raw === 'boolean') {
              return raw ? 1 : 0;
            }
            return Number(raw);
          });
          datasets.push({
            label: def.label,
            data: values,
            borderColor: def.color,
            backgroundColor: def.color,
            tension: 0.25,
            fill: false,
            yAxisID: def.axis || 'y',
            spanGaps: true,
            _formatter: def.asPercent ? formatPercent : null,
          });
        });

        const scales = {
          y: { beginAtZero: true, suggestedMax: 1, title: { display: true, text: 'Score' } },
        };
        if (datasets.some((ds) => ds.yAxisID === 'y1')) {
          scales.y1 = { position: 'right', beginAtZero: true, suggestedMax: 1, grid: { drawOnChartArea: false }, title: { display: true, text: 'Binary / counts' } };
        }

        if (evalDetailChart) {
          evalDetailChart.destroy();
        }
        evalDetailChart = new Chart(canvas.getContext('2d'), {
          type: 'line',
          data: { labels, datasets },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            stacked: false,
            scales,
            plugins: {
              legend: { position: 'bottom' },
              tooltip: {
                callbacks: {
                  label: function(context) {
                    const datasetDef = datasets[context.datasetIndex];
                    const value = context.parsed.y;
                    if (value === null || value === undefined) {
                      return context.dataset.label + ': n/a';
                    }
                    if (datasetDef._formatter) {
                      return context.dataset.label + ': ' + datasetDef._formatter(value);
                    }
                    return context.dataset.label + ': ' + value;
                  },
                },
              },
            },
          },
        });
      }

      function setupAntagonistDetail(antData) {
        const select = document.getElementById('antRunSelect');
        if (!select) {
          return;
        }
        select.innerHTML = '';
        if (!antData || !antData.run_ids || antData.run_ids.length === 0) {
          const opt = document.createElement('option');
          opt.value = '';
          opt.textContent = 'No antagonist runs yet';
          select.appendChild(opt);
          select.disabled = true;
          renderAntDetailChart(null);
          renderAntFlagTable([]);
          return;
        }
        select.disabled = false;
        antData.run_ids.forEach((runId, idx) => {
          const ts = antData.labels && antData.labels[idx] ? new Date(antData.labels[idx]).toLocaleString() : runId;
          const opt = document.createElement('option');
          opt.value = runId;
          opt.textContent = ts + ' (' + runId + ')';
          select.appendChild(opt);
        });
        select.onchange = (event) => {
          loadAntRunDetail(event.target.value);
        };
        if (select.value) {
          loadAntRunDetail(select.value);
        } else {
          renderAntDetailChart(null);
          renderAntFlagTable([]);
        }
      }

      async function loadAntRunDetail(runId) {
        if (!runId) {
          renderAntDetailChart(null);
          renderAntFlagTable([]);
          return;
        }
        try {
          const response = await fetch('/manifest?run_id=' + encodeURIComponent(runId));
          if (!response.ok) {
            throw new Error('manifest lookup failed (' + response.status + ')');
          }
          const manifest = await response.json();
          const summary = (manifest.payload && manifest.payload.summary) || null;
          renderAntDetailChart(summary);
          const telemetry = (summary && summary.flag_telemetry) || [];
          renderAntFlagTable(telemetry);
        } catch (error) {
          console.error('Failed to load antagonist detail', error);
          renderAntDetailChart(null);
          renderAntFlagTable([]);
        }
      }

      function renderAntDetailChart(summary) {
        const canvas = document.getElementById('antDetailChart');
        const empty = document.getElementById('antDetailEmpty');
        const summaryEl = document.getElementById('antRunSummary');
        if (!canvas || !empty || !summaryEl) {
          return;
        }
        if (!summary || !summary.severity_breakdown) {
          canvas.style.display = 'none';
          empty.style.display = 'block';
          summaryEl.textContent = '';
          if (antDetailChart) {
            antDetailChart.destroy();
            antDetailChart = null;
          }
          return;
        }
        const total = summary.total_records || 0;
        const flagged = summary.flagged_records || 0;
        const rateText = summary.flag_rate !== undefined ? formatPercent(summary.flag_rate) : 'n/a';
        summaryEl.textContent = `Flag rate: ${rateText} (${flagged}/${total} records)`;
        canvas.style.display = 'block';
        empty.style.display = 'none';
        const labels = ['HIGH', 'MEDIUM', 'LOW'];
        const severity = summary.severity_breakdown || {};
        const values = labels.map((label) => Number(severity[label] || 0));
        if (antDetailChart) {
          antDetailChart.destroy();
        }
        antDetailChart = new Chart(canvas.getContext('2d'), {
          type: 'bar',
          data: {
            labels,
            datasets: [
              {
                label: 'Flag Count',
                data: values,
                backgroundColor: ['#ff7f0e', '#9467bd', '#7f7f7f'],
              },
            ],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              y: { beginAtZero: true, precision: 0, ticks: { stepSize: 1 } },
            },
            plugins: {
              legend: { display: false },
            },
          },
        });
      }

      function renderAntFlagTable(flags) {
        const table = document.getElementById('antFlagTable');
        const empty = document.getElementById('antFlagEmpty');
        if (!table || !empty) {
          return;
        }
        const tbody = table.querySelector('tbody');
        if (!tbody) {
          return;
        }
        tbody.innerHTML = '';
        if (!flags || flags.length === 0) {
          table.style.display = 'none';
          empty.style.display = 'block';
          return;
        }
        table.style.display = 'table';
        empty.style.display = 'none';
        flags.forEach((flag) => {
          const row = document.createElement('tr');
          const metrics = flag.metrics || {};
          row.innerHTML = `
            <td>${flag.claim_id ?? '-'}</td>
            <td>${flag.severity ?? '-'}</td>
            <td>${metrics.chirality_score !== undefined && metrics.chirality_score !== null ? metrics.chirality_score.toFixed(3) : '-'}</td>
            <td>${metrics.entailment_score !== undefined && metrics.entailment_score !== null ? metrics.entailment_score.toFixed(3) : '-'}</td>
            <td>${metrics.citation_valid === false ? 'invalid' : 'valid'}</td>
            <td>${(flag.issues || []).map((issue) => issue.issue_type).join(', ') || '-'}</td>
          `;
          tbody.appendChild(row);
        });
      }

      document.addEventListener('DOMContentLoaded', initCharts);
    </script>
  </body>
</html>
"""
        html = (
            html_template
            .replace("__INDEX_PATH__", str(INDEX_PATH.relative_to(REPO_ROOT)))
            .replace("__RUNS_TABLE__", _render_runs_table(runs))
        )
        encoded = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def serve(port: int = DEFAULT_PORT) -> None:
    server = HTTPServer(("127.0.0.1", port), DashboardHandler)
    print(f"[dashboard] Serving Thinker metrics at http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - manual stop
        print("\n[dashboard] shutting down...")
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Thinker metric manifests")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Listening port")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    serve(port=args.port)


def _collect_metrics_payload() -> Dict[str, Any]:
    manifests = list(_iter_manifests())
    return {
        "train": _build_stage_series(manifests, "train", TRAIN_METRICS, _extract_train_metrics),
        "evaluation": _build_stage_series(manifests, "eval", EVAL_METRICS, _extract_eval_metrics),
        "antagonist": _build_stage_series(manifests, "antagonist", ANTAGONIST_METRICS, _extract_antagonist_metrics),
    }


def write_json(handler: BaseHTTPRequestHandler, payload: Dict[str, Any], status: int = 200) -> None:
    encoded = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def _build_stage_series(manifests: List[Dict[str, Any]], stage: str, metric_keys: List[str], extractor) -> Dict[str, Any]:
    if not manifests:
        return _empty_series(metric_keys)
    stage_manifests = [m for m in manifests if m.get("stage") == stage]
    if not stage_manifests:
        return _empty_series(metric_keys)
    stage_manifests.sort(key=lambda m: m.get("timestamp", ""))
    labels = []
    run_ids = []
    series = {key: [] for key in metric_keys}
    for manifest in stage_manifests:
        labels.append(manifest.get("timestamp"))
        run_ids.append(manifest.get("run_id"))
        metrics = extractor(manifest)
        for key in metric_keys:
            series[key].append(metrics.get(key))
    return {"labels": labels, "series": series, "run_ids": run_ids}


def _empty_series(keys: List[str]) -> Dict[str, Any]:
    return {"labels": [], "series": {key: [] for key in keys}, "run_ids": []}


def _extract_train_metrics(manifest: Dict[str, Any]) -> Dict[str, Any]:
    payload = manifest.get("payload") or {}
    metrics = payload.get("metrics") or {}
    final_loss = metrics.get("final_loss")
    citation_rate = None
    if "citation_validation" in metrics:
        citation_rate = metrics["citation_validation"].get("invalid_rate")
    training_report = metrics.get("training_report")
    if training_report:
        if final_loss is None:
            final_loss = training_report.get("final_loss")
        if citation_rate is None:
            citation_data = training_report.get("citation_validation") or {}
            citation_rate = citation_data.get("invalid_rate")
    return {
        "final_loss": _safe_float(final_loss),
        "citation_invalid_rate": _safe_float(citation_rate),
    }


def _extract_eval_metrics(manifest: Dict[str, Any]) -> Dict[str, Any]:
    payload = manifest.get("payload") or {}
    metrics = payload.get("metrics") or {}
    return {key: _safe_float(metrics.get(key)) for key in EVAL_METRICS}


def _extract_antagonist_metrics(manifest: Dict[str, Any]) -> Dict[str, Any]:
    payload = manifest.get("payload") or {}
    summary = payload.get("summary") or {}
    flagged = summary.get("flagged_records")
    total = summary.get("total_records")
    flag_rate = None
    if flagged is not None and total:
        try:
            flag_rate = float(flagged) / float(total)
        except (TypeError, ValueError, ZeroDivisionError):
            flag_rate = None
    severity = summary.get("severity_breakdown") or {}
    return {
        "flag_rate": flag_rate,
        "flagged_records": _safe_float(flagged),
        "high_flags": _safe_float(severity.get("HIGH")),
        "medium_flags": _safe_float(severity.get("MEDIUM")),
    }


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
