from collections import namedtuple
from collections.abc import Iterable
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm

from data.genx_utils.labels import SparselyBatchedObjectLabels
from data.genx_utils.sequence_rnd import SequenceForRandomAccess
from data.genx_utils.liosam_sequence import build_liosam_sequences, LiosamSequenceDataset
from data.utils.augmentor import RandomSpatialAugmentorGenX
from data.utils.types import DatasetMode, LoaderDataDictGenX, DatasetType, DataType


class SequenceDataset(Dataset):
    def __init__(self,
                 path: Path,
                 dataset_mode: DatasetMode,
                 dataset_config: Dict[str, Any]):
        assert path.is_dir()

        ### extract settings from config ###
        sequence_length = dataset_config['sequence_length']
        assert isinstance(sequence_length, int)
        assert sequence_length > 0
        self.output_seq_len = sequence_length

        ev_representation_name = dataset_config['ev_repr_name']
        downsample_by_factor_2 = dataset_config['downsample_by_factor_2']
        only_load_end_labels = dataset_config['only_load_end_labels']
        load_depth = dataset_config.get('load_depth', False)  # Default to False for backwards compatibility

        augm_config = dataset_config.get('data_augmentation', {})

        ####################################
        dataset_name = dataset_config['name']
        if dataset_name == 'virtual':
            dataset_type = DatasetType.VIRTUAL
        elif dataset_name == 'dsec':
            dataset_type = DatasetType.DSEC
        else:
            raise NotImplementedError(f"Dataset '{dataset_name}' not supported. Only 'virtual' and 'dsec' are available.")
        depth_range = dataset_config.get('depth_range', {})
        min_depth = float(depth_range.get('min', 0.1))
        max_depth = float(depth_range.get('max', 100.0))
        self.sequence = SequenceForRandomAccess(path=path,
                                                ev_representation_name=ev_representation_name,
                                                sequence_length=sequence_length,
                                                dataset_type=dataset_type,
                                                downsample_by_factor_2=downsample_by_factor_2,
                                                only_load_end_labels=only_load_end_labels,
                                                load_depth=load_depth,
                                                min_depth=min_depth,
                                                max_depth=max_depth)

        # 仅在训练模式下启用随机水平翻转
        self.spatial_augmentor = None
        if dataset_mode == DatasetMode.TRAIN:
            resolution_hw = tuple(dataset_config['resolution_hw'])
            assert len(resolution_hw) == 2
            ds_by_factor_2 = dataset_config['downsample_by_factor_2']
            if ds_by_factor_2:
                resolution_hw = tuple(x // 2 for x in resolution_hw)

            # data_augmentation 允许为嵌套 dict：{"random": {...}, "stream": {...}}
            if isinstance(augm_config, dict):
                random_augm_cfg = augm_config.get('random', {})
            else:
                random_augm_cfg = {}

            self.spatial_augmentor = RandomSpatialAugmentorGenX(
                dataset_hw=resolution_hw,
                automatic_randomization=True,
                augm_config=random_augm_cfg,
            )

    def only_load_labels(self):
        self.sequence.only_load_labels()

    def load_everything(self):
        self.sequence.load_everything()

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index: int) -> LoaderDataDictGenX:

        item = self.sequence[index]

        if self.spatial_augmentor is not None and not self.sequence.is_only_loading_labels():
            item = self.spatial_augmentor(item)

        return item


class CustomConcatDataset(ConcatDataset):
    datasets: List[SequenceDataset]

    def __init__(self, datasets: Iterable[SequenceDataset]):
        super().__init__(datasets=datasets)

    def only_load_labels(self):
        for idx, dataset in enumerate(self.datasets):
            self.datasets[idx].only_load_labels()

    def load_everything(self):
        for idx, dataset in enumerate(self.datasets):
            self.datasets[idx].load_everything()


def build_random_access_dataset(dataset_mode: DatasetMode, dataset_config: Dict[str, Any]) -> CustomConcatDataset:
    dataset_path = Path(dataset_config['path'])
    assert dataset_path.is_dir(), f'{str(dataset_path)}'

    dataset_name = dataset_config.get('name', 'dsec')
    if dataset_name == 'liosam':
        _, _, train_rnd, val_rnd = build_liosam_sequences(
            dataset_path,
            dataset_config,
            train_ratio=dataset_config.get('train_ratio', 0.9),
        )
        rnd_list = train_rnd if dataset_mode == DatasetMode.TRAIN else val_rnd

        # 如果配置了 debug_start_npz_index，并且当前是训练集，则只保留“起始 npz 最接近该下标”的一个序列，
        # 用于单 batch 过拟合调试（配合 --debug_fixed_batch 使用）。
        debug_start_idx = dataset_config.get('debug_start_npz_index', None)
        if debug_start_idx is not None and dataset_mode == DatasetMode.TRAIN and len(rnd_list) > 0:
            try:
                target = int(debug_start_idx)
                best_seq = None
                best_dist = None
                for seq in rnd_list:
                    # 每个 LiOSAM 序列都有 frame_indices 属性：是 index.txt 中的行号列表
                    start_idx = int(seq.frame_indices[0]) if len(seq.frame_indices) > 0 else 0
                    dist = abs(start_idx - target)
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_seq = seq
                if best_seq is not None:
                    rnd_list = [best_seq]
                    print(f'[liosam][debug] 使用 debug_start_npz_index={target} 对应的序列，'
                          f'实际序列起始 npz 索引={best_seq.frame_indices[0]}')
            except Exception as e:
                print(f'[liosam][debug] 解析 debug_start_npz_index 出错，忽略该选项: {debug_start_idx}, error={e}')

        seq_datasets = [LiosamSequenceDataset(seq) for seq in rnd_list]
        print(f'[liosam] random access {dataset_mode.name}: {len(seq_datasets)} sequences')
        return CustomConcatDataset(seq_datasets)

    mode2str = {DatasetMode.TRAIN: 'train',
                DatasetMode.VALIDATION: 'val',
                DatasetMode.TESTING: 'test'}

    split_path = dataset_path / mode2str[dataset_mode]
    assert split_path.is_dir()

    seq_datasets = list()
    for entry in tqdm(split_path.iterdir(), desc=f'creating rnd access {mode2str[dataset_mode]} datasets'):
        seq_datasets.append(SequenceDataset(path=entry, dataset_mode=dataset_mode, dataset_config=dataset_config))

    return CustomConcatDataset(seq_datasets)


def get_weighted_random_sampler(dataset: CustomConcatDataset) -> WeightedRandomSampler:
    class2count = dict()
    ClassAndCount = namedtuple('ClassAndCount', ['class_ids', 'counts'])
    classandcount_list = list()
    print('--- START generating weighted random sampler ---')
    dataset.only_load_labels()
    for idx, data in enumerate(tqdm(dataset, desc='iterate through dataset')):
        labels: SparselyBatchedObjectLabels = data[DataType.OBJLABELS_SEQ]
        label_list, valid_batch_indices = labels.get_valid_labels_and_batch_indices()
        class_ids_seq = list()
        for label in label_list:
            class_ids_numpy = np.asarray(label.class_id.numpy(), dtype='int32')
            class_ids_seq.append(class_ids_numpy)
        class_ids_seq, counts_seq = np.unique(np.concatenate(class_ids_seq), return_counts=True)
        for class_id, count in zip(class_ids_seq, counts_seq):
            class2count[class_id] = class2count.get(class_id, 0) + count
        classandcount_list.append(ClassAndCount(class_ids=class_ids_seq, counts=counts_seq))
    dataset.load_everything()

    class2weight = {}
    for class_id, count in class2count.items():
        count = max(count, 1)
        class2weight[class_id] = 1 / count

    weights = []
    for classandcount in classandcount_list:
        weight = 0
        for class_id, count in zip(classandcount.class_ids, classandcount.counts):
            # Not only weight depending on class but also depending on number of occurrences.
            # This will bias towards sampling "frames" with more bounding boxes.
            weight += class2weight[class_id] * count
        weights.append(weight)

    print('--- DONE generating weighted random sampler ---')
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
