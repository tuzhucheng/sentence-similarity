"""
Driver program for training and evaluation.
"""
import argparse

import torch.nn as nn

from datasets.sick import SICK
from models.sentence_embedding_baseline import SmoothInverseFrequencyBaseline


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentence similarity models')
    parser.add_argument('--alpha', type=float, default=1e-3, help='Smoothing term for smooth inverse frequency baseline model')
    args = parser.parse_args()

    train_loader, dev_loader, test_loader = SICK.iters(shuffle=False)

    embedding_dim = SICK.TEXT.vocab.vectors.size()
    embedding = nn.Embedding(embedding_dim[0], embedding_dim[1])
    embedding.weight = nn.Parameter(SICK.TEXT.vocab.vectors)

    model = SmoothInverseFrequencyBaseline(args.alpha, embedding)
    model.fit(train_loader)
    train_pearson, train_spearman = model.score(train_loader)

    print(f"Training set pearson coefficient is {train_pearson} and spearman coefficient is {train_spearman}")
