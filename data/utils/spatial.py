from typing import Dict, Any

from data.utils.types import DatasetType

_type_2_hw = {
    DatasetType.VIRTUAL: (480, 640),  # Virtual/Synthetic dataset
    DatasetType.DSEC: (480, 640),     # DSEC monocular event camera
    DatasetType.LIOSAM: (260, 346),   # LiOSAM odometry2 event tensor 2*260*346
}

_str_2_type = {
    'virtual': DatasetType.VIRTUAL,
    'dsec': DatasetType.DSEC,
    'liosam': DatasetType.LIOSAM,
}


def get_original_hw(dataset_type: DatasetType):
    return _type_2_hw[dataset_type]


def get_dataloading_hw(dataset_config: Dict[str, Any]):
    dataset_name = dataset_config['name']
    hw = get_original_hw(dataset_type=_str_2_type[dataset_name])
    downsample_by_factor_2 = dataset_config['downsample_by_factor_2']
    if downsample_by_factor_2:
        hw = tuple(x // 2 for x in hw)
    return hw
