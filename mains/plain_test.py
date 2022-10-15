# encoding: utf-8

import argparse, os, sys, time, shutil, torch
sys.path.append('.')

from config import cfg
from utils.filesys import generate_output_dir
from utils.logger import setup_logger
from models import build_model
from data import build_data_loader
from functions import build_function
from solver import build_optimizer
from engine.plain_inference import do_inference


def test():
    model = build_model(cfg)
    model.load_state_dict(torch.load(cfg.TEST.WEIGHT).cpu().state_dict())
    test_loader = build_data_loader(cfg, is_train=False)

    do_inference(cfg, model, test_loader)

def main():
    parser = argparse.ArgumentParser(description="PyTorch Training Template")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    else:
        raise Exception("Input config file, please!")
    cfg.merge_from_list(args.opts)

    generate_output_dir(cfg)
    logger = setup_logger(cfg, 0)
    logger.info("Running with config:\n{}".format(cfg))
    cfg.freeze()

    test()

if __name__ == "__main__":
    main()
