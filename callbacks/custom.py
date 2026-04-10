from typing import Dict, Any
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint

from callbacks.depth_viz import DepthVizCallback


def get_ckpt_callback(config: Dict[str, Any]) -> ModelCheckpoint:
    """
    Create checkpoint callback for the model
    
    Args:
        config: Configuration
    
    Returns:
        ModelCheckpoint callback
    """
    model_name = config['model']['name']

    prefix = 'val'
    if model_name == 'depth':
        # For depth estimation, monitor RMSE (lower is better)
        metric = 'rmse'
        mode = 'min'
    else:
        raise NotImplementedError(f"Model '{model_name}' not supported. Only 'depth' is available.")
    
    ckpt_callback_monitor = prefix + '/' + metric
    filename_monitor_str = prefix + '_' + metric

    ckpt_filename = 'epoch={epoch:03d}-step={step}-' + filename_monitor_str + '={' + ckpt_callback_monitor + ':.2f}'
    cktp_callback = ModelCheckpoint(
        monitor=ckpt_callback_monitor,
        filename=ckpt_filename,
        auto_insert_metric_name=False,  # because backslash would create a directory
        save_top_k=1,
        mode=mode,
        every_n_epochs=config['logging']['ckpt_every_n_epochs'],
        save_last=True,
        verbose=True)
    cktp_callback.CHECKPOINT_NAME_LAST = 'last_epoch={epoch:03d}-step={step}'
    return cktp_callback


def get_viz_callback(config: Dict[str, Any]) -> Callback:
    """
    Create visualization callback for the model
    
    Args:
        config: Configuration
    
    Returns:
        Visualization callback
    """
    model_name = config['model']['name']

    if model_name == 'depth':
        return DepthVizCallback(config=config)
    else:
        raise NotImplementedError(f"Model '{model_name}' not supported. Only 'depth' is available.")
