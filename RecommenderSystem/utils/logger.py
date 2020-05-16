import os
import sys
import glob
import time
import shutil
import logging

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def create_logger(cfg):
    log_path = os.path.join(cfg.EXP.LOG_PATH, cfg.EXP.NAME, f"{cfg.EXP.NAME}-{time.strftime('%Y%m%d-%H%M%S')}")
    create_exp_dir(log_path, scripts_to_save=None)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_path, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)

    return logger, log_path