# encoding: utf-8

import os, datetime

def mkdir_if_not_exist(path_list):
    all_path = os.path.join(*path_list)
    if not os.path.exists(all_path):
        os.makedirs(all_path)
    return all_path

def generate_output_dir(cfg):
    if cfg.RECORD.RESULT_DIR : return
    cfg.RECORD.RESULT_DIR = mkdir_if_not_exist([
        cfg.RECORD.OUTPUT_DIR, 
        cfg.TASK.NAME,
        datetime.datetime.now().strftime("%YY_%mM_%dD_%HH_%MM_%SS_%f")
    ])

def record_config_file(cfg):
    with open(os.path.join(cfg.RECORD.RESULT_DIR, 'config.yaml'), 'w') as f:
        f.write('' + str(cfg))
