# encoding: utf-8
import os
import logging
import time
import yaml

__author__ = 'jeffye'

fmt_full = logging.Formatter("%(asctime)s %(pathname)s %(filename)s %(funcName)s %(lineno)s \
      %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")

fmt_module = logging.Formatter("%(asctime)s %(filename)s %(funcName)s %(lineno)s \
      %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")


def get_logger(name, level=logging.DEBUG, fmt=fmt_module, output='output.log'):
    logger = logging.getLogger(name)
    logger.propagate = False
    rht = logging.FileHandler(output, mode='a')
    rht.setFormatter(fmt)

    logging.basicConfig()
    logging.StrFormatStyle = fmt

    import sys
    chlr = logging.StreamHandler(stream=sys.stderr)
    chlr.setFormatter(fmt)

    logger.addHandler(rht)
    logger.addHandler(chlr)
    logger.setLevel(level)

    return logger


def setup_logging(
        default_path='configs/logging.yaml',
        default_level=logging.INFO,
        env_key='LOG_CFG',
        add_time_stamp=False
):
    """Setup logging configuration
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.load(f.read())
        if add_time_stamp:
            add_time_2_log_filename(config)
        # logging.config.dictConfig(config)
        # logging.basicConfig(path)
        logging.basicConfig(level=default_level)
    else:
        logging.basicConfig(level=default_level)


def add_time_2_log_filename(config):
    for k, v in config.iteritems():
        if k == 'filename':
            config[k] = v + "." + time.strftime("%Y-%d-%m-%s")
            print('log file name: %s' % config[k])
        elif type(v) is dict:
            add_time_2_log_filename(v)


def goal_prompt(logger, prompt='What are you testing in this experiment? '):
    print("            ***************************")
    goal = input(prompt)
    logger.info("            ***************************")
    logger.info("TEST GOAL: %s" % goal)


def log_git_commit(logger):
    try:
        commit = get_git_revision_hash()
        logger.info("current git commit: %s" % commit)
    except:
        logger.info('cannot get git commit.')


def get_git_revision_hash():
    import subprocess
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])
