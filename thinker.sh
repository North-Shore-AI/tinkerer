#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
VENV_DIR="$ROOT_DIR/.venv"
REQ_FILE="$ROOT_DIR/requirements.txt"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"
DEFAULT_CONFIG="thinker/configs/pipeline_scifact.yaml"
DEBUG_CONFIG="thinker/configs/pipeline_scifact_debug.yaml"

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
Recommended flow (per run): 5 → 1 → 3 → 4 (data setup → validate → train → eval)
Use options 8/9 for quick debug loops.

1) Validate (SciFact config)
2) Train (HF PEFT, full config)
3) Train (Tinker backend, full config)
4) Evaluate (SciFact config)
5) Data setup – SciFact (with embedding validation)
6) Data setup – FEVER
7) Custom Thinker command
8) Train HF (SciFact debug config: 32 samples, 1 epoch)
9) Train Tinker (SciFact debug config: 32 samples, 1 epoch)
10) Quit
HELP
}

run_validate() { run_cli --config "$DEFAULT_CONFIG" validate; }
run_train_hf() { run_cli --config "$DEFAULT_CONFIG" train --backend hf_peft; }
run_train_tinker() { run_cli --config "$DEFAULT_CONFIG" train --backend tinker; }
run_eval() { run_cli --config "$DEFAULT_CONFIG" eval; }
run_data_scifact() { run_cli data setup --dataset scifact --validation-mode embedding --similarity-threshold 0.7; }
run_data_fever() { run_cli data setup --dataset fever --skip-validation; }
run_custom() {
  read -rp "Enter Thinker args (after 'python -m thinker.cli'): " line
  if [[ -n "$line" ]]; then
    run_cli $line
  fi
}
run_train_hf_debug() { run_cli --config "$DEBUG_CONFIG" train --backend hf_peft; }
run_train_tinker_debug() { run_cli --config "$DEBUG_CONFIG" train --backend tinker; }

main() {
  ensure_python
  ensure_venv
  ensure_requirements
  source "$VENV_DIR/bin/activate"
  while true; do
    usage
    read -rp "Select option [1-10]: " choice
    case "$choice" in
      1) run_validate ;;
      2) run_train_hf ;;
      3) run_train_tinker ;;
      4) run_eval ;;
      5) run_data_scifact ;;
      6) run_data_fever ;;
      7) run_custom ;;
      8) run_train_hf_debug ;;
      9) run_train_tinker_debug ;;
      10) echo "Goodbye."; exit 0 ;;
      *) echo "Invalid choice" ;;
    esac
    echo ""
  done
}

main
