POETRY ?= poetry
PYTHON ?= python

.PHONY: install fmt lint test embed cluster weak validate train costs

install:
	$(POETRY) install --sync

fmt:
	$(POETRY) run black src tests
	$(POETRY) run isort src tests

lint:
	$(POETRY) run mypy src/worldvue

test:
	$(POETRY) run pytest

embed:
	$(POETRY) run worldvue embed --input data/articles.parquet

cluster:
	$(POETRY) run worldvue cluster --min-size 3

weak:
	$(POETRY) run worldvue weak label --all

validate:
	$(POETRY) run worldvue validate --sample 100 --budget 0.10

train:
	$(POETRY) run worldvue train style --method hybrid

costs:
	$(POETRY) run worldvue costs show
