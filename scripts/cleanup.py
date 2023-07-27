#!/usr/bin/env python
import shutil
import os
import wandb
import logging

logger = logging.getLogger(__name__)
api = wandb.Api()

for run in api.runs("henri-iser/ettcl-trash"):
    if 'output_dir' in run.config:
        print(0)
        dir_to_delete = run.config['output_dir']
        if os.path.exists(dir_to_delete):
            logger.warning(f"Run {run.name} ({run.id}): deleting directory '{dir_to_delete}'")
            shutil.rmtree(dir_to_delete)
            run.delete(delete_artifacts=True)
    elif 'training' in run.config:
        print(2)
        if 'output_dir' in run.config['training']:
            dir_to_delete = run.config['training']['output_dir']
            if os.path.exists(dir_to_delete):
                logger.warning(f"Run {run.name} ({run.id}): deleting directory '{dir_to_delete}'")
                shutil.rmtree(dir_to_delete)
                run.delete(delete_artifacts=True)
    elif 'config' in run.config:
        print(1)
        if 'output_dir' in run.config['config']:
            dir_to_delete = run.config['config']['output_dir']
            if os.path.exists(dir_to_delete):
                logger.warning(f"Run {run.name} ({run.id}): deleting directory '{dir_to_delete}'")
                shutil.rmtree(dir_to_delete)
                run.delete(delete_artifacts=True)
