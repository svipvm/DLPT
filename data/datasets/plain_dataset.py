from torchvision.datasets import ImageFolder

def get_dataset_from_folder(cfg, transformer, is_train=True):
    if is_train:
        return ImageFolder(cfg.DATASETS.TRAIN.PATH[0], transform=transformer)
    else:
        return ImageFolder(cfg.DATASETS.TEST.PATH[0], transform=transformer)

