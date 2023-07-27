# .EXPORT_ALL_VARIABLES:
# TRANSFORMERS_VERBOSITY=info
# WANDB_MODE=disabled
# WANDB_DISABLED=true

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
	./scripts/cleanup.py
