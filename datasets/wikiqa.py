"""
Data comes from Microsoft Research WikiQA Corpus
https://www.microsoft.com/en-us/download/details.aspx?id=52419
"""
import numpy as np
import torch
from torchtext.data import BucketIterator, Field, interleave_keys, RawField
from torchtext.data.dataset import TabularDataset
from torchtext.data.pipeline import Pipeline


def get_class_probs(sim, *args):
    """
    Convert a single label into class probabilities.
    """
    class_probs = np.zeros(WikiQA.num_classes)
    class_probs[0], class_probs[1] = 1 - sim, sim

    return class_probs


class WikiQA(TabularDataset):

    name = 'wikiqa'
    num_classes = 2

    def __init__(self, path, format, fields, skip_header=True, **kwargs):
        super(WikiQA, self).__init__(path, format, fields, skip_header, **kwargs)

        # We want to keep a raw copy of the sentence for some models and for debugging
        RAW_TEXT_FIELD = RawField()
        for ex in self.examples:
            raw_sentence_a, raw_sentence_b = ex.sentence_a[:], ex.sentence_b[:]
            setattr(ex, 'raw_sentence_a', raw_sentence_a)
            setattr(ex, 'raw_sentence_b', raw_sentence_b)

        self.fields['raw_sentence_a'] = RAW_TEXT_FIELD
        self.fields['raw_sentence_b'] = RAW_TEXT_FIELD

    @staticmethod
    def sort_key(ex):
        return interleave_keys(
            len(ex.sentence_a), len(ex.sentence_b))

    @classmethod
    def splits(cls, text_field, label_field, id_field, path='data/wikiqa', root='', train='train/WikiQA-train.tsv',
               validation='dev/WikiQA-dev.tsv', test='test/WikiQA-test.tsv', **kwargs):

        return super(WikiQA, cls).splits(path, root, train, validation, test,
                                           format='tsv',
                                           fields=[('id', id_field), ('sentence_a', text_field),
                                                   ('docid', None), ('document_title', None),
                                                   ('sid', id_field), ('sentence_b', text_field),
                                                   ('relatedness_score', label_field)],
                                           skip_header=True)

    @classmethod
    def iters(cls, batch_size=64, device=-1, shuffle=True, vectors='glove.840B.300d'):
        cls.TEXT = Field(sequential=True, tokenize='spacy', lower=True, batch_first=True)
        cls.LABEL = Field(sequential=False, use_vocab=False, batch_first=True, tensor_type=torch.FloatTensor, postprocessing=Pipeline(get_class_probs))
        cls.ID = RawField()

        train, val, test = cls.splits(cls.TEXT, cls.LABEL, cls.ID)

        cls.TEXT.build_vocab(train, vectors=vectors)

        return BucketIterator.splits((train, val, test), batch_size=batch_size, shuffle=shuffle, repeat=False, device=device)
