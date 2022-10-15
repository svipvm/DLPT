# encoding: utf-8

from torch.utils import data

from .datasets import get_dataset_from_folder
from .transforms import build_transform


def build_data_loader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.DATASETS.TRAIN.BATCH_SIZE
        shuffle = True
    else:
        batch_size = cfg.DATASETS.TEST.BATCH_SIZE
        shuffle = False

    transforms = build_transform(cfg, is_train)
    datasets = __get_dataset(cfg, transforms, is_train)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return data_loader

def __get_dataset(cfg, transforms, is_train=True):
    return get_dataset_from_folder(cfg, transforms, is_train)
