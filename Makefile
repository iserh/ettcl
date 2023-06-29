SHELL := /bin/bash

.DEFAULT: help
help:
	@echo "make test"
	@echo "    run tests using pytest"
	@echo "make format"
	@echo "    use isort, reindent & black to format source code"
	@echo "make lint"
	@echo "    use flake8 to lint source code"

install:
	pip install -r requirements.txt

test:
	pytest

format:
	isort $$(find ettcl/ tests/ scripts/ -type f -name '*.py')
	reindent -r -n ettcl/ tests/ scripts/
	black ettcl/ tests/ scripts/

lint:
	flake8 $$(find ettcl/ tests/ scripts/ -type f -name '*.py')

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} \;
	find . -type d -name .ipynb_checkpoints -prune -exec rm -rf {} \;
	find . -type d -name *.egg-info -prune -exec rm -rf {} \;

finetune:
	TRANSFORMERS_VERBOSITY=info \
	./scripts/finetune.py configs/finetune_config.yml
