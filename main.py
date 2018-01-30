"""
Driver program for training and evaluation.
"""
import argparse

import torch.nn as nn
import torch.nn.functional as F

from datasets.sick import SICK
from models.sentence_embedding_baseline import SmoothInverseFrequencyBaseline


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentence similarity models')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--unsupervised', action='store_true', default=False, help='Set this flag to use unsupervised mode.')
    parser.add_argument('--alpha', type=float, default=1e-3, help='Smoothing term for smooth inverse frequency baseline model')
    parser.add_argument('--no-remove-special-direction', action='store_true', default=False, help='Set to not remove projection onto first principal component')
    parser.add_argument('--frequency-dataset', default='enwiki', choices=['train', 'enwiki'])
    args = parser.parse_args()

    args.supervised = not args.unsupervised
    args.remove_special_direction = not args.no_remove_special_direction

    train_loader, dev_loader, test_loader = SICK.iters(batch_size=args.batch_size, shuffle=False)

    embedding_dim = SICK.TEXT.vocab.vectors.size()
    embedding = nn.Embedding(embedding_dim[0], embedding_dim[1])
    embedding.weight = nn.Parameter(SICK.TEXT.vocab.vectors)
    embedding.weight.requires_grad = False

    model = SmoothInverseFrequencyBaseline(5, args.alpha, embedding,
                                           remove_special_direction=args.remove_special_direction,
                                           frequency_dataset=args.frequency_dataset,
                                           supervised=args.supervised)

    model.populate_word_frequency_estimation(train_loader)

    if args.supervised:
        # TODO implement
        pass

    train_pearson, train_spearman = model.score(train_loader)
    dev_pearson, dev_spearman = model.score(dev_loader)
    test_pearson, test_spearman = model.score(test_loader)

    print(f"Training set pearson coefficient is {train_pearson:.4} and spearman coefficient is {train_spearman:.4}")
    print(f"Dev set pearson coefficient is {dev_pearson:.4} and spearman coefficient is {dev_spearman:.4}")
    print(f"Testing set pearson coefficient is {test_pearson:.4} and spearman coefficient is {test_spearman:.4}")
