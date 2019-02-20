import logging
import datetime
import sys
import os


def setup_logger(name):
    now = datetime.datetime.now().strftime('%m%d%H%M%S')
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    if not os.path.exists("./log"):
        os.mkdir("./log")
    handler = logging.FileHandler('log/{}.log'.format(now), mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger
