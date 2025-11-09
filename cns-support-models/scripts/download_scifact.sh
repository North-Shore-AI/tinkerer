#!/usr/bin/env bash
# Download the public SciFact dataset tarball and unpack it into data/raw/scifact.
# Usage: bash scripts/download_scifact.sh [output_dir]
set -euo pipefail

OUT_DIR="${1:-cns-support-models/data/raw/scifact}"
URL="https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"

mkdir -p "${OUT_DIR}"
echo "[download] Fetching SciFact data into ${OUT_DIR}"

curl -L "${URL}" | tar -xz -C "${OUT_DIR}" --strip-components=1

echo "[done] Files available under ${OUT_DIR}"
