import os
import shutil

import torch

import utils


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        name, ext = os.path.splitext(filename)
        best_filename = name + '_best' + ext
        shutil.copyfile(filename, best_filename)
