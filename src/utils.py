import logging
import sys

def get_logger():
    logger = logging.getLogger("DL_Project")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # STDOUT-ra írunk, amit a Docker a log fájlba irányít
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger



