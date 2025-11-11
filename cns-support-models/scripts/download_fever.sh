#!/usr/bin/env bash
# Download FEVER dataset and wiki dump from the official release.
# Usage: bash scripts/download_fever.sh [output_dir]
set -euo pipefail

OUT_DIR="${1:-cns-support-models/data/raw/fever}"
ZENODO_BASE="https://zenodo.org/api/files/91ad9fca-c15e-4069-80f9-bac6ec5c2d9b"

mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

download_file() {
    local filename="$1"
    local url="${ZENODO_BASE}/${filename}"
    echo "[download] Fetching ${filename}"
    curl -L -C - -o "${filename}" "${url}"
}

download_file "train.jsonl"
download_file "shared_task_dev.jsonl"
download_file "shared_task_dev_public.jsonl"
download_file "shared_task_test.jsonl"
download_file "paper_dev.jsonl"
download_file "paper_test.jsonl"

download_file "wiki-pages.zip"
echo "[extract] Unzipping wiki pages"
mkdir -p wiki-pages
unzip -o wiki-pages.zip -d wiki-pages

echo "[done] FEVER files available under ${OUT_DIR}"
