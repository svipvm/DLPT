# encoding: utf-8


import logging, os, sys

class Logger:
    def __init__(self, cfg):
        self.logger_name = "-".join([cfg.MODEL.NAME, cfg.DATASETS.NAME])
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(logging.DEBUG)
        
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        save_file_name = os.path.join(cfg.RECORD.RESULT_DIR, "log.txt")
        fh = logging.FileHandler(save_file_name, mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def get_logger(self):
        return self.logger

def setup_logger(cfg, distributed_rank):
    if distributed_rank > 0:
        return logger
    logger = Logger(cfg).get_logger()
    return logger

def get_current_logger(cfg):
    logger_name = "-".join([cfg.MODEL.NAME, cfg.DATASETS.NAME])
    return logging.getLogger(logger_name)