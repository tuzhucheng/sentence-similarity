import torch.nn as nn

from datasets.sick import SICK
from datasets.wikiqa import WikiQA


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
