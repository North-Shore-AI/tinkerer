#!/bin/bash
# Regenerate SciFact training data with document citations

set -euo pipefail

echo "[regen] Regenerating SciFact training data with document citations..."

python3 cns-support-models/scripts/convert_scifact.py     --claims_json cns-support-models/data/raw/scifact/claims_train.jsonl     --corpus_json cns-support-models/data/raw/scifact/corpus.jsonl     --out cns-support-models/data/processed/scifact_claim_extractor.jsonl

echo "[regen] Done! Training data now includes document citations."
echo "[regen] Example output:"
head -1 cns-support-models/data/processed/scifact_claim_extractor.jsonl | python3 -c "import sys, json; obj = json.load(sys.stdin); print('COMPLETION:', obj['completion'][:300])"
