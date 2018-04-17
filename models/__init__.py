import numpy as np

from models.sentence_embedding_baseline import SmoothInverseFrequencyBaseline
from models.mpcnn import MPCNN
from models.mpcnn_lite import MPCNNLite
from models.bimpm import BiMPM


def get_model(args, dataset_cls, embedding):
    if args.model == 'sif':
        args.supervised = not args.unsupervised
        args.remove_special_direction = not args.no_remove_special_direction
        model = SmoothInverseFrequencyBaseline(dataset_cls.num_classes, args.alpha, embedding,
                                               remove_special_direction=args.remove_special_direction,
                                               frequency_dataset=args.frequency_dataset,
                                               supervised=args.supervised)
    elif args.model == 'mpcnn':
        model = MPCNN(embedding, 300, 300, 20, [1, 2, 3, np.inf], 150, dataset_cls.num_classes, 0.5)
    elif args.model == 'mpcnn-lite':
        model = MPCNNLite(embedding, 300, 300, [1, 2, 3, np.inf], 150, dataset_cls.num_classes, 0.5)
    elif args.model == 'bimpm':
        model = BiMPM(embedding, 300, 50, 20, 100, dataset_cls.num_classes, 0.1)
    else:
        raise ValueError(f'Unrecognized dataset: {args.model}')

    if args.device != -1:
        model = model.cuda()

    return model
