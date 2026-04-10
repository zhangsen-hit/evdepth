from functools import partialmethod
from pathlib import Path
from typing import List, Union, Dict, Any

from torch.utils.data import Dataset
from tqdm import tqdm

from data.genx_utils.sequence_for_streaming import SequenceForIter, RandAugmentIterDataPipe
from data.genx_utils.liosam_sequence import build_liosam_sequences
from data.utils.stream_concat_datapipe import ConcatStreamingDataPipe
from data.utils.stream_sharded_datapipe import ShardedStreamingDataPipe
from data.utils.types import DatasetMode, DatasetType


def build_streaming_dataset(dataset_mode: DatasetMode, dataset_config: Dict[str, Any], batch_size: int, num_workers: int) \
        -> Union[ConcatStreamingDataPipe, ShardedStreamingDataPipe]:
    dataset_path = Path(dataset_config['path'])
    assert dataset_path.is_dir(), f'{str(dataset_path)}'

    dataset_name = dataset_config.get('name', 'dsec')
    if dataset_name == 'liosam':
        train_stream, val_stream, _, _ = build_liosam_sequences(
            dataset_path,
            dataset_config,
            train_ratio=dataset_config.get('train_ratio', 0.9),
        )
        datapipes = train_stream if dataset_mode == DatasetMode.TRAIN else val_stream
        if not datapipes:
            raise RuntimeError(
                f'[liosam] No {dataset_mode.name} sequences. '
                'Check index.txt and that enough contiguous 5ms±3ms frames exist for sequence_length.'
            )
        print(f'[liosam] streaming {dataset_mode.name}: {len(datapipes)} sequences')
    else:
        mode2str = {DatasetMode.TRAIN: 'train',
                    DatasetMode.VALIDATION: 'val',
                    DatasetMode.TESTING: 'test'}

        split_path = dataset_path / mode2str[dataset_mode]
        assert split_path.is_dir()
        datapipes = list()
        num_full_sequences = 0
        num_splits = 0
        num_split_sequences = 0
        guarantee_labels = dataset_mode == DatasetMode.TRAIN
        for entry in tqdm(split_path.iterdir(), desc=f'creating streaming {mode2str[dataset_mode]} datasets'):
            new_datapipes = get_sequences(path=entry, dataset_config=dataset_config, guarantee_labels=guarantee_labels)
            if len(new_datapipes) == 1:
                num_full_sequences += 1
            else:
                num_splits += 1
                num_split_sequences += len(new_datapipes)
            datapipes.extend(new_datapipes)
        print(f'{num_full_sequences=}\n{num_splits=}\n{num_split_sequences=}')

    if dataset_mode == DatasetMode.TRAIN:
        return build_streaming_train_dataset(
            datapipes=datapipes, dataset_config=dataset_config, batch_size=batch_size, num_workers=num_workers)
    elif dataset_mode in (DatasetMode.VALIDATION, DatasetMode.TESTING):
        return build_streaming_evaluation_dataset(datapipes=datapipes, batch_size=batch_size)
    else:
        raise NotImplementedError


def get_sequences(path: Path, dataset_config: Dict[str, Any], guarantee_labels: bool) -> List[SequenceForIter]:
    assert path.is_dir()

    ### extract settings from config ###
    sequence_length = dataset_config['sequence_length']
    ev_representation_name = dataset_config['ev_repr_name']
    downsample_by_factor_2 = dataset_config['downsample_by_factor_2']
    load_depth = dataset_config.get('load_depth', False)  # Default to False for backwards compatibility
    
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
    ####################################
    if guarantee_labels:
        return SequenceForIter.get_sequences_with_guaranteed_labels(
            path=path,
            ev_representation_name=ev_representation_name,
            sequence_length=sequence_length,
            dataset_type=dataset_type,
            downsample_by_factor_2=downsample_by_factor_2,
            load_depth=load_depth,
            min_depth=min_depth,
            max_depth=max_depth)
    return [SequenceForIter(
        path=path,
        ev_representation_name=ev_representation_name,
        sequence_length=sequence_length,
        dataset_type=dataset_type,
        downsample_by_factor_2=downsample_by_factor_2,
        load_depth=load_depth,
        min_depth=min_depth,
        max_depth=max_depth)]


def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return NewCls


def build_streaming_train_dataset(datapipes: List[Dataset],
                                  dataset_config: Dict[str, Any],
                                  batch_size: int,
                                  num_workers: int) -> ConcatStreamingDataPipe:
    assert len(datapipes) > 0
    augmentation_datapipe_type = partialclass(RandAugmentIterDataPipe, dataset_config=dataset_config)
    streaming_dataset = ConcatStreamingDataPipe(datapipe_list=datapipes,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                augmentation_pipeline=augmentation_datapipe_type,
                                                print_seed_debug=False)
    return streaming_dataset


def build_streaming_evaluation_dataset(datapipes: List[Dataset],
                                       batch_size: int) -> ShardedStreamingDataPipe:
    assert len(datapipes) > 0
    fill_value = datapipes[0].get_fully_padded_sample()
    streaming_dataset = ShardedStreamingDataPipe(datapipe_list=datapipes, batch_size=batch_size, fill_value=fill_value)
    return streaming_dataset
