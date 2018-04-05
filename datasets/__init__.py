import torch.nn as nn

from datasets.sick import SICK


def get_dataset(args):
    if args.dataset == 'sick':
        train_loader, dev_loader, test_loader = SICK.iters(batch_size=args.batch_size, shuffle=True)

        embedding_dim = SICK.TEXT.vocab.vectors.size()
        embedding = nn.Embedding(embedding_dim[0], embedding_dim[1])
        embedding.weight = nn.Parameter(SICK.TEXT.vocab.vectors)
        embedding.weight.requires_grad = False

        return SICK, train_loader, dev_loader, test_loader, embedding
    else:
        raise ValueError(f'Unrecognized dataset: {args.dataset}')
