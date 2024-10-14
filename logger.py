import logging
import os
from termcolor import colored
import sys
import functools

class false_logger:
    """An expedient."""

    def __init__(self, **kwargs):
        pass

    @staticmethod
    def info(msg):
        print(msg)

    @staticmethod
    def warn(msg):
        print(msg)

    @staticmethod
    def warning(msg):
        print(msg)


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=None):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)
        name = '_' + name if name is not None else name
        file_handler = logging.FileHandler(os.path.join(output_dir, f'log{name}.log'), mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)

    return logger