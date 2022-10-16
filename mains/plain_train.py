# encoding: utf-8

import argparse, os, sys
sys.path.append('.')

from config import cfg
from utils.filesys import *
from utils.logger import setup_logger
from models import build_model
from data import build_data_loader
from functions import build_function
from solver import build_optimizer
from engine.plain_trainer import do_train


def train():
    model = build_model(cfg)
    train_loader = build_data_loader(cfg, is_train=True)
    valid_loader = build_data_loader(cfg, is_train=False)
    optimizer = build_optimizer(cfg, model)
    loss_fn = build_function(cfg)

    do_train(cfg, model, train_loader, valid_loader, optimizer, loss_fn)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Training Template")
    parser.add_argument("--config_file", default="configs/plain_config.yaml", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    generate_output_dir(cfg)
    logger = setup_logger(cfg, 0)
    
    record_config_file(cfg)
    logger.info("Running with config:{}".format(cfg))
    cfg.freeze()

    train()
    logger.info("This result was saved to: {}".format(cfg.RECORD.RESULT.DIR))

if __name__ == "__main__":
    main()
