import torch


def save_checkpoint(state, filename):
    torch.save(state, filename)
