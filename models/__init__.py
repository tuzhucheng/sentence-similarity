from models.sentence_embedding_baseline import SmoothInverseFrequencyBaseline


def get_model(args, dataset_cls, embedding):
    if args.model == 'sif':
        args.supervised = not args.unsupervised
        args.remove_special_direction = not args.no_remove_special_direction
        model = SmoothInverseFrequencyBaseline(dataset_cls.num_classes, args.alpha, embedding,
                                               remove_special_direction=args.remove_special_direction,
                                               frequency_dataset=args.frequency_dataset,
                                               supervised=args.supervised)
    else:
        raise ValueError(f'Unrecognized dataset: {args.model}')

    if args.device != -1:
        model = model.cuda()

    return model
