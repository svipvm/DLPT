# encoding: utf-8

import torch.nn.functional as F

def build_function(cfg):
    if cfg.LOSS.NAME == 'CrossEntropy':
        return F.cross_entropy
    
    raise Exception("Not found this functions!")