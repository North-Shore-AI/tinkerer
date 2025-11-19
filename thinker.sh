#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
VENV_DIR="$ROOT_DIR/.venv"
REQ_FILE="$ROOT_DIR/requirements.txt"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"
DEFAULT_CONFIG="thinker/configs/pipeline_scifact.yaml"
DEBUG_CONFIG="thinker/configs/pipeline_scifact_debug.yaml"
LIMITED_CONFIG="thinker/configs/pipeline_scifact_limited.yaml"
MICRO_CONFIG="thinker/configs/pipeline_scifact_micro.yaml"

ensure_python() {
  if ! command -v python3 >/dev/null 2>&1; then
    echo "[thinker] python3 not found. Please install Python 3 before running this script."
    exit 1
  fi
}

ensure_venv() {
  if [ ! -d "$VENV_DIR" ]; then
    echo "[thinker] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
  fi
}

ensure_requirements() {
  if [ -f "$REQ_FILE" ]; then
    echo "[thinker] Installing requirements..."
    "$PIP_BIN" install --upgrade pip >/dev/null
    "$PIP_BIN" install -r "$REQ_FILE" >/dev/null
  else
    echo "[thinker] WARNING: requirements.txt not found; skipping automatic dependency install."
  fi
}

run_cli() {
  "$PYTHON_BIN" -m thinker.cli "$@"
}

usage() {
  cat <<'HELP'
Thinker Command Menu
====================
Recommended flow (per run): 10 → 1 → 2/3/4 → 5 (data setup → validate → train → eval)
Use debug / limited options for smoke tests and diagnostics.

 1) Validate (SciFact config)
 2) Train (HF PEFT, full config)
 3) Train (Tinker backend, full config)
 4) Train (Tinker micro config, 5 samples)
 5) Evaluate (SciFact config)
 6) Evaluate (Limited config, 5 samples)
 7) Run full pipeline (HF backend)
 8) Run full pipeline (Tinker backend)
 9) Run Antagonist on latest eval
10) Data setup – SciFact (with embedding validation)
11) Data setup – FEVER
12) Show pipeline info
13) Show latest Tinker manifest
14) Train HF (SciFact debug config: 32 samples, 1 epoch)
15) Train Tinker (SciFact debug config: 32 samples, 1 epoch)
16) Custom Thinker command
17) Launch dashboard server (serve_dashboard)
18) Launch dashboard manager (menu wrapper)
19) Quit
HELP
}

run_validate() { run_cli --config "$DEFAULT_CONFIG" validate; }
run_train_hf() { run_cli --config "$DEFAULT_CONFIG" train --backend hf_peft; }
run_train_tinker() { run_cli --config "$DEFAULT_CONFIG" train --backend tinker; }
run_eval() { run_cli --config "$DEFAULT_CONFIG" eval; }
run_eval_limited() { run_cli --config "$LIMITED_CONFIG" eval --skip-validation; }
run_data_scifact() { run_cli data setup --dataset scifact --validation-mode embedding --similarity-threshold 0.7; }
run_data_fever() { run_cli data setup --dataset fever --skip-validation; }
run_pipeline_hf() { run_cli --config "$DEFAULT_CONFIG" run --backend hf_peft; }
run_pipeline_tinker() { run_cli --config "$DEFAULT_CONFIG" run --backend tinker; }
run_info() { run_cli --config "$DEFAULT_CONFIG" info; }
run_manifest() { run_cli --config "$DEFAULT_CONFIG" manifest; }
run_custom() {
  read -rp "Enter Thinker args (after 'python -m thinker.cli'): " line
  if [[ -n "$line" ]]; then
    run_cli $line
  fi
}
run_train_hf_debug() { run_cli --config "$DEBUG_CONFIG" train --backend hf_peft; }
run_train_tinker_debug() { run_cli --config "$DEBUG_CONFIG" train --backend tinker; }
run_antagonist() { run_cli --config "$DEFAULT_CONFIG" antagonist; }
run_train_tinker_micro() { run_cli --config "$MICRO_CONFIG" train --backend tinker; }
run_dashboard_server() { "$PYTHON_BIN" scripts/serve_dashboard.py --venv "$VENV_DIR" --port 43117; }
run_dashboard_manager() { "$PYTHON_BIN" scripts/dashboard_manager.py; }

main() {
  ensure_python
  ensure_venv
  ensure_requirements
  source "$VENV_DIR/bin/activate"
  while true; do
    usage
    read -rp "Select option [1-19]: " choice
    case "$choice" in
      1) run_validate ;;
      2) run_train_hf ;;
      3) run_train_tinker ;;
      4) run_train_tinker_micro ;;
      5) run_eval ;;
      6) run_eval_limited ;;
      7) run_pipeline_hf ;;
      8) run_pipeline_tinker ;;
      9) run_antagonist ;;
      10) run_data_scifact ;;
      11) run_data_fever ;;
      12) run_info ;;
      13) run_manifest ;;
      14) run_train_hf_debug ;;
      15) run_train_tinker_debug ;;
      16) run_custom ;;
      17) run_dashboard_server ;;
      18) run_dashboard_manager ;;
      19) echo "Goodbye."; exit 0 ;;
      *) echo "Invalid choice" ;;
    esac
    echo ""
  done
}

main
