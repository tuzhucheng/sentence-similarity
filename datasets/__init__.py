import torch
from torch.autograd import Variable
import torch.nn as nn

from datasets.sick import SICK
from datasets.wikiqa import WikiQA

from metrics.retrieval_metrics import MAP, MRR
from metrics.pearson_correlation import PearsonCorrelation
from metrics.spearman_correlation import SpearmanCorrelation


def get_dataset(args):
    if args.dataset == 'sick':
        train_loader, dev_loader, test_loader = SICK.iters(batch_size=args.batch_size, device=args.device, shuffle=True)

        embedding_dim = SICK.TEXT.vocab.vectors.size()
        embedding = nn.Embedding(embedding_dim[0], embedding_dim[1])
        embedding.weight = nn.Parameter(SICK.TEXT.vocab.vectors)
        embedding.weight.requires_grad = False

        return SICK, train_loader, dev_loader, test_loader, embedding
    elif args.dataset == 'wikiqa':
        train_loader, dev_loader, test_loader = WikiQA.iters(batch_size=args.batch_size, device=args.device, shuffle=True)

        embedding_dim = WikiQA.TEXT.vocab.vectors.size()
        embedding = nn.Embedding(embedding_dim[0], embedding_dim[1])
        embedding.weight = nn.Parameter(WikiQA.TEXT.vocab.vectors)
        embedding.weight.requires_grad = False

        return WikiQA, train_loader, dev_loader, test_loader, embedding
    else:
        raise ValueError(f'Unrecognized dataset: {args.dataset}')


def get_dataset_configurations(args):
    if args.dataset == 'sick':
        loss_fn = nn.KLDivLoss()
        metrics = {
            'pearson': PearsonCorrelation(),
            'spearman': SpearmanCorrelation()
        }

        if args.unsupervised:
            args.epochs = 0

        def y_to_score(y, batch):
            num_classes = batch.relatedness_score.size(1)
            predict_classes = Variable(torch.arange(1, num_classes + 1).expand(len(batch.id), num_classes))
            if y.is_cuda:
                with torch.cuda.device(y.get_device()):
                    predict_classes = predict_classes.cuda()

            return (predict_classes * y).sum(dim=1)

        def resolved_pred_to_score(y, batch):
            num_classes = batch.relatedness_score.size(1)
            predict_classes = Variable(torch.arange(1, num_classes + 1).expand(len(batch.id), num_classes))
            if y.is_cuda:
                with torch.cuda.device(y.get_device()):
                    predict_classes = predict_classes.cuda()

            return (predict_classes * y.exp()).sum(dim=1)

        resolved_pred_to_score = (lambda y, batch: y) if args.unsupervised else resolved_pred_to_score

        return loss_fn, metrics, y_to_score, resolved_pred_to_score

    elif args.dataset == 'wikiqa':
        # Always supervised
        loss_fn = nn.KLDivLoss()
        metrics = {
            'map': MAP(),
            'mrr': MRR()
        }
        if args.unsupervised:
            args.epochs = 0

        def resolved_pred_to_score(y, batch):
            num_classes = batch.relatedness_score.size(1)
            predict_classes = Variable(torch.arange(0, num_classes).expand(len(batch.id), num_classes))
            if y.is_cuda:
                with torch.cuda.device(y.get_device()):
                    predict_classes = predict_classes.cuda()

            return (predict_classes * y.exp()).sum(dim=1)

        def y_to_score(y, batch):
            return y[:, 1]

        return loss_fn, metrics, y_to_score, resolved_pred_to_score
