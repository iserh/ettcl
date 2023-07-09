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

mcc-evaluate-colbert:
	python scripts/multiclass/evaluate_colbert.py configs/multiclass/eval_config_colbert.yml

mcc-evaluate-sbert:
	python scripts/multiclass/evaluate_sbert.py configs/multiclass/eval_config_sbert.yml

mcc-evaluate-scolbert:
	python scripts/multiclass/evaluate_scolbert.py configs/multiclass/eval_config_scolbert.yml

mcc-evaluate-fasttext:
	python scripts/multiclass/evaluate_fasttext.py configs/multiclass/eval_config_fasttext.yml

mcc-finetune-baseline:
	python scripts/multiclass/finetune_baseline.py configs/multiclass/finetune_config_baseline.yml

mcc-finetune-colbert:
	python scripts/multiclass/finetune_colbert.py configs/multiclass/finetune_config_colbert.yml

mcc-finetune-sbert:
	python scripts/multiclass/finetune_sbert.py configs/multiclass/finetune_config_sbert.yml

mcc-finetune-scolbert:
	python scripts/multiclass/finetune_scolbert.py configs/multiclass/finetune_config_scolbert.yml


mlc-evaluate-colbert:
	python scripts/multilabel/evaluate_colbert.py configs/multilabel/eval_config_colbert.yml

mlc-evaluate-sbert:
	python scripts/multilabel/evaluate_sbert.py configs/multilabel/eval_config_sbert.yml

mlc-evaluate-scolbert:
	python scripts/multilabel/evaluate_scolbert.py configs/multilabel/eval_config_scolbert.yml

mlc-evaluate-fasttext:
	python scripts/multilabel/evaluate_fasttext.py configs/multilabel/eval_config_fasttext.yml

# mlc-finetune-baseline:
# 	python scripts/multilabel/finetune_baseline.py configs/multilabel/finetune_config_baseline.yml

mlc-finetune-colbert:
	python scripts/multilabel/finetune_colbert.py configs/multilabel/finetune_config_colbert.yml

mlc-finetune-sbert:
	python scripts/multilabel/finetune_sbert.py configs/multilabel/finetune_config_sbert.yml

mlc-finetune-scolbert:
	python scripts/multilabel/finetune_scolbert.py configs/multilabel/finetune_config_scolbert.yml

