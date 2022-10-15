# encoding: utf-8


import sys, unittest
sys.path.append('.')

from config import cfg
from data import build_data_loader
from models import build_model
from solver import build_optimizer
from functions import build_function


class TestDataSet(unittest.TestCase):

    def test_dataset(self):
        cfg.merge_from_file("configs/plain_config.yaml")
        print('open this config file:\n{}'.format(cfg))
        
        # train_loader = build_data_loader(cfg, True)
        # print(train_loader)
        # test_loader = build_data_loader(cfg, False)
        # print(test_loader)

        # model = build_model(cfg)
        # print(model)

        # optimizer = build_optimizer(cfg, model)
        # print(optimizer)

        # loss_fn = build_function(cfg)
        # print(loss_fn)

        # from IPython import embed;
        # embed()


if __name__ == '__main__':
    unittest.main()
