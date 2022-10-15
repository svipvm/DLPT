# encoding: utf-8

import torchvision.transforms as T

def build_transform(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.NORM_MEAN, std=cfg.INPUT.NORM_STD)
    if is_train:
        transform = T.Compose([T.Resize(40),
            T.RandomResizedCrop(32, (0.64, 1), (1.0, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize_transform])
    else:
        transform =  T.Compose([T.ToTensor(), normalize_transform])
    return transform
