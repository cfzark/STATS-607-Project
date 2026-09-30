PYTHON ?= python
RESULTS ?= results

export MPLBACKEND := Agg
export OPENBLAS_NUM_THREADS := 1
export OMP_NUM_THREADS := 1
export MKL_NUM_THREADS := 1

.DEFAULT_GOAL := help
.PHONY: help reproduce test plot validate

help:
	@echo "make reproduce  Run all 8 configurations, plot CSV results, and validate outputs"
	@echo "make test       Run automated tests"
	@echo "make plot       Regenerate figures from existing CSVs and validate outputs"
	@echo "make validate   Check all saved outputs and write validation.json"

reproduce:
	$(PYTHON) scripts/runner.py --all --output-dir "$(RESULTS)"

test:
	$(PYTHON) -m pytest -q

plot:
	$(PYTHON) scripts/runner.py --all --stage plot --output-dir "$(RESULTS)"

validate:
	$(PYTHON) scripts/runner.py --all --stage validate --output-dir "$(RESULTS)"
