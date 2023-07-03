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

evaluate-colbert:
	./scripts/evaluate_colbert.py configs/eval_config_colbert.yml

evaluate-sbert:
	./scripts/evaluate_sbert.py configs/eval_config_sbert.yml

finetune-colbert:
	./scripts/finetune_colbert.py configs/finetune_config_colbert.yml

finetune-sbert:
	./scripts/finetune_sbert.py configs/finetune_config_sbert.yml
