# encoding: utf-8

from .resnet import resnet18

def build_model(cfg):
    device = 'cuda' if cfg.MODEL.DEVICES is not None else 'cpu'
    if cfg.MODEL.NAME == 'ResNet18':
        return resnet18().to(device)
    
    raise Exception("Not found this modle!")