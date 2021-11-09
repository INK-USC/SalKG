import csv
import logging

import numpy as np
from pytorch_lightning.utilities.distributed import rank_zero_only


def cal_2hop_rel_emb(rel_emb):
    n_rel = rel_emb.shape[0]
    u, v = np.meshgrid(np.arange(n_rel), np.arange(n_rel))
    expanded = rel_emb[v.reshape(-1)] + rel_emb[u.reshape(-1)]
    return np.concatenate([rel_emb, expanded], 0)


def load_qa_predictions(path):
    preds = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row_arr = np.array([float(x) for x in row])
            preds.append(row_arr[1:])
    return np.stack(preds)


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def freeze_net(module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_net(module):
    for p in module.parameters():
        p.requires_grad = True
