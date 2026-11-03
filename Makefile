.PHONY: install data baseline eval results test lint notebooks site all \
        local-setup local-check local-paraphrase local-baselines local-train local-eval local-awq local-push local-all

install:
	pip install -e ".[dev,serve,space]"

data:
	jobx schema && jobx build-data

baseline:
	jobx regex-baseline

eval:
	jobx eval && jobx taxonomy > /dev/null

results:
	jobx results-table

notebooks:
	python scripts/build_notebooks.py && python scripts/render_modelfile.py

lint:
	ruff check . && ruff format --check . && mypy

test:
	pytest

site:
	jobx site

all: data baseline eval results lint test

# ---------------------------------------------------------------------------------------------------------
# Local GPU (RTX 4080 laptop, WSL2). Guide: docs/local_gpu_setup.md. Same steps and outputs as the notebooks.
# Override the profile with  make local-train PROFILE=configs/gpu/my_gpu.yaml ; extra flags via ARGS="...".
PY       ?= .venv/bin/python
AWQ_PY   ?= .venv-awq/bin/python
PROFILE  ?= local_rtx4080
ARGS     ?=
LOCAL     = $(PY) -m jobx.local

local-setup:              ## create .venv with the pinned GPU stack (~5 min, ~6 GB download)
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -r requirements/local-gpu.txt -e ".[hub,api,dev]"

local-check:              ## ~1 min: environment + GPU sanity check (PASS/FAIL table)
	$(LOCAL) check $(ARGS)

local-paraphrase:         ## ~25-35 min: notebook 01 (7B paraphrases + real-posting drafts)
	$(LOCAL) paraphrase --profile $(PROFILE) $(ARGS)

local-baselines:          ## ~60-80 min: notebook 02 (zero/few-shot, free + guided, 1.5B and 7B)
	$(LOCAL) baselines --profile $(PROFILE) $(ARGS)

local-train:              ## ~30-45 min: notebook 03 (QLoRA, bf16, resumes from checkpoints)
	$(LOCAL) train --profile $(PROFILE) $(ARGS)

local-eval:               ## ~15-20 min: notebook 04 (fine-tune predictions + merge)
	$(LOCAL) eval --profile $(PROFILE) $(ARGS)

.venv-awq:
	uv venv --python 3.11 .venv-awq
	uv pip install --python $(AWQ_PY) -r requirements/local-awq.txt -e ".[hub]"

local-awq: .venv-awq      ## ~20-30 min: notebook 05 (AWQ + 16-bit/8-bit/AWQ benchmark) in its own venv
	$(AWQ_PY) -m jobx.local check --requirements requirements/local-awq.txt --no-bnb || \
	  echo "AWQ environment check reported problems; continuing (the AWQ row stays pending if it fails)"
	$(AWQ_PY) -m jobx.local awq --profile $(PROFILE) $(ARGS)

local-push:               ## ~5-10 min: upload results + adapter/merged/AWQ to the Hub, like the notebooks
	$(LOCAL) push --profile $(PROFILE) $(ARGS)

local-all:                ## ~3-4 h: everything in order, then a summary
	$(LOCAL) all --profile $(PROFILE) $(ARGS)
	$(MAKE) local-awq PROFILE=$(PROFILE)
	$(LOCAL) push --profile $(PROFILE)
	jobx eval && jobx results-table
	@echo "Done. Commit results/ + README.md, then run the quantize-benchmark workflow for the GGUF rows."
