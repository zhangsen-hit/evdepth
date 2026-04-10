from pathlib import Path
from typing import Union, Dict, Any

import wandb

from loggers.wandb_logger import WandbLogger


def get_wandb_logger(full_config: Dict[str, Any]) -> WandbLogger:
    wandb_config = full_config['wandb']
    wandb_runpath = wandb_config.get('wandb_runpath')

    if wandb_runpath is None:
        wandb_id = wandb.util.generate_id()
        print(f'new run: generating id {wandb_id}')
    else:
        wandb_id = Path(wandb_runpath).name
        print(f'using provided id {wandb_id}')

    logger = WandbLogger(
        project=wandb_config['project_name'],
        group=wandb_config['group_name'],
        wandb_id=wandb_id,
        log_model=True,
        save_last_only_final=False,
        save_code=True,
        config_args=full_config,
    )

    return logger


def get_ckpt_path(logger: WandbLogger, wandb_config: Dict[str, Any]) -> Union[Path, None]:
    artifact_name = wandb_config.get('artifact_name')
    assert artifact_name is not None, 'Artifact name is required to resume from checkpoint.'
    print(f'resuming checkpoint from artifact {artifact_name}')
    artifact_local_file = wandb_config.get('artifact_local_file')
    if artifact_local_file is not None:
        artifact_local_file = Path(artifact_local_file)
    if isinstance(logger, WandbLogger):
        resume_path = logger.get_checkpoint(
            artifact_name=artifact_name,
            artifact_filepath=artifact_local_file)
    else:
        resume_path = artifact_local_file
    assert resume_path.exists()
    assert resume_path.suffix == '.ckpt', resume_path.suffix
    return resume_path
