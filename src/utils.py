import logging
import sys
import torch

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

def compute_class_weights(y_train, device):
    counts = torch.bincount(y_train)
    counts = torch.max(counts, torch.ones_like(counts))
    total = len(y_train)
    num_classes = len(counts)
    weights = total / (num_classes * counts.float())
    return weights.to(device)