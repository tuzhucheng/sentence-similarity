"""
Implementation of "A Simple but Tough-to-Beat Baseline for Sentence Embeddings"
https://openreview.net/forum?id=SyK00v5xx
"""
from collections import defaultdict

from sklearn.decomposition import TruncatedSVD
import torch
from torch.autograd import Variable
import torch.nn as nn


class SmoothInverseFrequencyBaseline(nn.Module):

    def __init__(self, n_out, alpha, embedding, remove_special_direction=True, frequency_dataset='enwiki', supervised=True):
        super(SmoothInverseFrequencyBaseline, self).__init__()
        self.n_out = n_out
        self.alpha = alpha
        self.embedding = embedding
        self.remove_special_direction = remove_special_direction
        self.frequency_dataset = frequency_dataset
        self.supervised = supervised

        self.unigram_prob = defaultdict(int)
        self.word_vec_dim = self.embedding.weight.size(1)

        self.classifier = nn.Sequential(
            nn.Linear(2*self.word_vec_dim, 2400),
            nn.Tanh(),
            nn.Linear(2400, n_out),
            nn.LogSoftmax(1)
        )

    def populate_word_frequency_estimation(self, data_loader):
        """
        Computing and storing the unigram probability.
        """
        if self.frequency_dataset == 'enwiki':
            with open('./data/enwiki_vocab_min200.txt') as f:
                for line in f:
                    word, freq = line.split(' ')
                    self.unigram_prob[word] = int(freq)
        else:
            for batch_idx, batch in enumerate(data_loader):
                for sent_a, sent_b in zip(batch.raw_sentence_a, batch.raw_sentence_b):
                    for w in sent_a:
                        self.unigram_prob[w] += 1

                    for w in sent_b:
                        self.unigram_prob[w] += 1

        total_words = sum(self.unigram_prob.values())
        for word, count in self.unigram_prob.items():
            self.unigram_prob[word] = count / total_words

    def _compute_sentence_embedding_as_weighted_sum(self, sentence, sentence_embedding):
        """
        Compute sentence embedding as weighted sum of word vectors of its individual words.
        :param sentence: Tokenized sentence
        :param sentence_embedding: A 2D tensor where dim 0 is the word vector dimension and dim 1 is the words
        :return: A vector that is the weighted word vectors of the words in the sentence
        """
        weights = [self.alpha / (self.unigram_prob.get(w, 0) + self.alpha) for w in sentence]
        weights.extend([0.0] * (sentence_embedding.size(1) - len(weights)))  # expand weights to cover padding
        weights = torch.FloatTensor(weights).expand_as(sentence_embedding.data)

        return (Variable(weights) * sentence_embedding).sum(1)

    def _remove_projection_on_first_principle_component(self, batch_sentence_embedding):
        """
        Remove the projection onto the first principle component of the sentences from each sentence embedding.
        See https://plot.ly/ipython-notebooks/principal-component-analysis/ for a nice tutorial on PCA.
        Follows official implementation at https://github.com/PrincetonML/SIF
        :param batch_sentence_embedding: A group of sentence embeddings (a 2D tensor, each row is a separate
        sentence and each column is a feature of a sentence)
        :return: A new batch sentence embedding with the projection removed
        """
        # Use truncated SVD to not center data
        svd = TruncatedSVD(n_components=1, n_iter=7)
        X = batch_sentence_embedding.data.numpy()
        svd.fit(X)
        pc = Variable(torch.FloatTensor(svd.components_))
        new_embedding = batch_sentence_embedding - batch_sentence_embedding.matmul(pc.transpose(0, 1)).matmul(pc)
        return new_embedding

    def cosine_similarity(self, sentence_embedding_a, sentence_embedding_b):
        """
        Compute cosine similarity of sentence embedding a and sentence embedding b.
        Each row is the sentence embedding (vector) for a sentence.
        """
        dot_product = (sentence_embedding_a * sentence_embedding_b).sum(1)
        norm_a = sentence_embedding_a.norm(p=2, dim=1)
        norm_b = sentence_embedding_b.norm(p=2, dim=1)
        cosine_sim = dot_product / (norm_a * norm_b)
        return cosine_sim

    def compute_sentence_embedding(self, batch):
        sentence_embedding_a = Variable(torch.zeros(batch.sentence_a.size(0), self.word_vec_dim))
        sentence_embedding_b = Variable(torch.zeros(batch.sentence_b.size(0), self.word_vec_dim))

        # compute weighted sum of word vectors
        for i, (raw_sent_a, raw_sent_b, sent_a_idx, sent_b_idx) in enumerate(zip(batch.raw_sentence_a, batch.raw_sentence_b, batch.sentence_a, batch.sentence_b)):
            sent_a = self.embedding(sent_a_idx).transpose(0, 1)
            sent_b = self.embedding(sent_b_idx).transpose(0, 1)

            sentence_embedding_a[i] = self._compute_sentence_embedding_as_weighted_sum(raw_sent_a, sent_a)
            sentence_embedding_b[i] = self._compute_sentence_embedding_as_weighted_sum(raw_sent_b, sent_b)

        # remove projection on first principle component
        if self.remove_special_direction:
            sentence_embedding_a = self._remove_projection_on_first_principle_component(sentence_embedding_a)
            sentence_embedding_b = self._remove_projection_on_first_principle_component(sentence_embedding_b)

        return sentence_embedding_a, sentence_embedding_b

    def forward(self, batch):
        if len(self.unigram_prob) == 0:
            raise ValueError('Word frequency lookup dictionary is not populated. Did you call populate_word_frequency_estimation?')

        sentence_embedding_a, sentence_embedding_b = self.compute_sentence_embedding(batch)
        if self.supervised:
            elem_wise_product = sentence_embedding_a * sentence_embedding_b
            abs_diff = torch.abs(sentence_embedding_a - sentence_embedding_b)
            concat_input = torch.cat([elem_wise_product, abs_diff], dim=1)
            scores = self.classifier(concat_input)

            # num_classes = batch.relatedness_score.size(1)
            # predict_classes = Variable(torch.arange(1, num_classes + 1).expand(len(batch.id), num_classes))
            # scores = (predict_classes * scores.exp()).sum(dim=1)
        else:
            scores = self.cosine_similarity(sentence_embedding_a, sentence_embedding_b)

        return scores
