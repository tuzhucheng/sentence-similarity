"""
Driver program for training and evaluation.
"""
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as O

from datasets import get_dataset
from models import get_model
import utils.utils as utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentence similarity models')
    parser.add_argument('--model', default='sif', choices=['sif'], help='Model to use')
    parser.add_argument('--dataset', default='sick', choices=['sick'], help='Dataset to use')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=1234, help='Seed for reproducibility')
    parser.add_argument('--device', type=int, default=0, help='Device, -1 for CPU')

    # Special options for SIF model
    parser.add_argument('--unsupervised', action='store_true', default=False, help='Set this flag to use unsupervised mode.')
    parser.add_argument('--alpha', type=float, default=1e-3, help='Smoothing term for smooth inverse frequency baseline model')
    parser.add_argument('--no-remove-special-direction', action='store_true', default=False, help='Set to not remove projection onto first principal component')
    parser.add_argument('--frequency-dataset', default='enwiki', choices=['train', 'enwiki'])

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.device != -1:
        torch.cuda.manual_seed(args.seed)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    dataset_cls, train_loader, dev_loader, test_loader, embedding = get_dataset(args)
    model = get_model(args, dataset_cls, embedding)

    loss_fn = nn.KLDivLoss()
    metrics = {
        'pearson':
    }
    optimizer = O.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=3e-4)

    if args.supervised:
        opt = O.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=3e-4)
        criterion = nn.KLDivLoss()
        best_dev_pearson = 0

        for epoch in range(1, args.epochs + 1):
            train_loader.init_epoch()
            model.train()
            for batch_idx, batch in enumerate(train_loader):
                opt.zero_grad()

                output = model(batch)
                loss = criterion(output, batch.relatedness_score)

                loss.backward()
                opt.step()

                if batch_idx % 10 == 0:
                    processed = min(batch_idx * args.batch_size, len(batch.dataset.examples))
                    logger.info(f'Train Epoch: {epoch} [{processed}/{len(batch.dataset.examples)} '
                                 f'({100. * batch_idx / (len(train_loader)):.0f}%)]\tLoss: {loss.data[0]:.6f}')

            model.eval()

            dev_loader.init_epoch()
            dev_loss = 0
            for batch_idx, batch in enumerate(dev_loader):
                output = model(batch)
                loss = criterion(output, batch.relatedness_score)

                if batch_idx % 10 == 0:
                    processed = min(batch_idx * args.batch_size, len(batch.dataset.examples))
                    logger.info(f'Dev Epoch: {epoch} [{processed}/{len(batch.dataset.examples)} '
                                 f'({100. * batch_idx / (len(dev_loader)):.0f}%)]\tLoss: {loss.data[0]:.6f}')

            train_pearson, train_spearman = model.score(train_loader)
            dev_pearson, dev_spearman = model.score(dev_loader)

            logger.info(f'Train pearson: {train_pearson:.4f}, spearman: {train_spearman:.4f}')
            logger.info(f'Dev pearson: {dev_pearson:.4f}, spearman: {dev_spearman:.4f}')

            if dev_pearson > best_dev_pearson:
                state_dict = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': opt.state_dict(),
                    'train_pearson': train_pearson,
                    'dev_pearson': dev_pearson
                }
                utils.save_checkpoint(state_dict, True, 'sick_supervised.model')
                best_dev_pearson = dev_pearson

        # Set model parameters to parameters that do best on dev set for evaluation
        checkpoint = torch.load('sick_supervised.model')
        model.load_state_dict(checkpoint['state_dict'])

    train_pearson, train_spearman = model.score(train_loader)
    dev_pearson, dev_spearman = model.score(dev_loader)
    test_pearson, test_spearman = model.score(test_loader)

    logger.info(f"Training set pearson coefficient is {train_pearson:.4} and spearman coefficient is {train_spearman:.4}")
    logger.info(f"Dev set pearson coefficient is {dev_pearson:.4} and spearman coefficient is {dev_spearman:.4}")
    logger.info(f"Testing set pearson coefficient is {test_pearson:.4} and spearman coefficient is {test_spearman:.4}")
