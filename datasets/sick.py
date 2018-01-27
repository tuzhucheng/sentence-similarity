"""
Data comes from SemEval-2014 Task 1: Evaluation of Compositional Distributional Semantic Models
on Full Sentences through Semantic Relatedness and Entailment
http://alt.qcri.org/semeval2014/task1/index.php?id=data-and-tools
"""
import torch
from torchtext.data import BucketIterator, Field, interleave_keys
from torchtext.data.dataset import TabularDataset


class SICK(TabularDataset):

    name = 'sick'

    @staticmethod
    def sort_key(ex):
        return interleave_keys(
            len(ex.sentence_a), len(ex.sentence_b))

    @classmethod
    def splits(cls, text_field, label_field, path='data/sick', root='', train='train/SICK_train.txt',
               validation='dev/SICK_trial.txt', test='test/SICK_test_annotated.txt', **kwargs):

        return super(SICK, cls).splits(path, root, train, validation, test,
                                       format='tsv',
                                       fields=[('id', label_field), ('sentence_a', text_field), ('sentence_b', text_field),
                                               ('relatedness_score', label_field), ('entailment', None)],
                                       skip_header=True)

    @classmethod
    def iters(cls, batch_size=64, device=0, vectors='glove.840B.300d'):

        TEXT = Field(sequential=True, tokenize='spacy', batch_first=True)
        LABEL = Field(sequential=False, use_vocab=False, batch_first=True, tensor_type=torch.FloatTensor)

        train, val, test = cls.splits(TEXT, LABEL)

        TEXT.build_vocab(train, vectors=vectors)

        return BucketIterator.splits((train, val, test), batch_size=batch_size, device=device)
