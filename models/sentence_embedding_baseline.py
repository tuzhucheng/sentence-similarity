"""
Implementation of "A Simple but Tough-to-Beat Baseline for Sentence Embeddings"
https://openreview.net/forum?id=SyK00v5xx
"""
from collections import defaultdict

import scipy.stats as stats
import torch


class SmoothInverseFrequencyBaseline(object):

    def __init__(self, alpha, embedding, remove_special_direction=True):
        self.alpha = alpha
        self.embedding = embedding
        self.remove_special_direction = remove_special_direction
        self.unigram_prob = defaultdict(int)

    def _compute_sentence_embedding_as_weighted_sum(self, sentence, sentence_embedding):
        """
        Compute sentence embedding as weighted sum of word vectors of its individual words.
        :param sentence: Tokenized sentence
        :param sentence_embedding: A 2D tensor where dim 0 is the word vector dimension and dim 1 is the words
        :return: A vector that is the weighted word vectors of the words in the sentence
        """
        weights = [self.alpha / (self.unigram_prob.get(w, 0) + self.alpha) for w in sentence]
        weights.extend([0.0] * (sentence_embedding.size(1) - len(weights)))  # expand weights to cover padding
        weights = torch.FloatTensor(weights).expand_as(sentence_embedding)

        return (weights * sentence_embedding).sum(1)


    def compute_sentence_embedding(self, batch):
        sentence_embedding_a = torch.zeros(batch.sentence_a.size(0), self.embedding.weight.size(1))
        sentence_embedding_b = torch.zeros(batch.sentence_b.size(0), self.embedding.weight.size(1))

        # compute weighted sum of word vectors
        for i, (raw_sent_a, raw_sent_b, sent_a_idx, sent_b_idx) in enumerate(zip(batch.raw_sentence_a, batch.raw_sentence_b, batch.sentence_a, batch.sentence_b)):
            sent_a = self.embedding(sent_a_idx).transpose(0, 1).data
            sent_b = self.embedding(sent_b_idx).transpose(0, 1).data

            sentence_embedding_a[i] = self._compute_sentence_embedding_as_weighted_sum(raw_sent_a, sent_a)
            sentence_embedding_b[i] = self._compute_sentence_embedding_as_weighted_sum(raw_sent_b, sent_b)

        # remove projection on first principle component
        if self.remove_special_direction:
            # TODO implement
            pass

        return sentence_embedding_a, sentence_embedding_b

    def fit(self, data_loader):
        """
        Training just consists of computing the unigram probability and getting the evaluation
        metrics on the training set.
        """
        for batch_idx, batch in enumerate(data_loader):
            for sent_a, sent_b in zip(batch.raw_sentence_a, batch.raw_sentence_b):
                for w in sent_a:
                    self.unigram_prob[w] += 1

                for w in sent_b:
                    self.unigram_prob[w] += 1

        total_words = sum(self.unigram_prob.values())
        for word, count in self.unigram_prob.items():
            self.unigram_prob[word] = count / total_words

    def score(self, data_loader):
        """
        Compute correlation between predicted score and actual score
        """
        scores = []
        gold = []
        for batch_idx, batch in enumerate(data_loader):
            sentence_embedding_a, sentence_embedding_b = self.compute_sentence_embedding(batch)

            dot_product = (sentence_embedding_a * sentence_embedding_b).sum(1)
            norm_a = sentence_embedding_a.norm(p=2, dim=1)
            norm_b = sentence_embedding_b.norm(p=2, dim=1)
            cosine_sim = dot_product / (norm_a * norm_b)
            scores.append(cosine_sim)
            gold.append(batch.relatedness_score.data)

        predicted_scores = torch.cat(scores).numpy()
        gold_scores = torch.cat(gold).numpy()

        pearson_score = stats.pearsonr(predicted_scores, gold_scores)[0]
        spearman_score = stats.spearmanr(predicted_scores, gold_scores)[0]

        return pearson_score, spearman_score
