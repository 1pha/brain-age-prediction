import time
import logging


def get_logger(name: str = None, filehandler: bool = False):
    name = name or __name__
    logging.basicConfig()
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)
    if filehandler:
        fname = f"{time.strftime('%Y%m%d-%H%M', time.localtime())}.log"
        logger.addHandler(logging.FileHandler(filename=fname))
    return logger