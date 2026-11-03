.PHONY: install data baseline eval results test lint notebooks site all

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
