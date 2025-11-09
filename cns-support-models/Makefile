PYTHON ?= python3
RAW_SCI := data/raw/scifact
PROC_SCI := data/processed/scifact_claim_extractor.jsonl

.PHONY: scifact clean

scifact: $(PROC_SCI)
	@echo "[make] SciFact processed dataset ready at $(PROC_SCI)"

$(RAW_SCI)/claims_train.jsonl:
	bash scripts/download_scifact.sh $(RAW_SCI)

$(PROC_SCI): $(RAW_SCI)/claims_train.jsonl $(RAW_SCI)/corpus.jsonl
	$(PYTHON) scripts/convert_scifact.py \
		--claims_json $(RAW_SCI)/claims_train.jsonl \
		--corpus_json $(RAW_SCI)/corpus.jsonl \
		--out $(PROC_SCI)

clean:
	rm -rf $(RAW_SCI) $(PROC_SCI)
