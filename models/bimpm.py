"""
This implementation is inspired by https://github.com/galsang/BIMPM-pytorch/blob/master/model/BIMPM.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiMPM(nn.Module):

    def __init__(self, word_embedding, n_word_dim, n_char_dim, n_perspectives, n_hidden_units, n_classes, dropout=0.1):
        super(BiMPM, self).__init__()
        self.word_embedding = word_embedding
        self.n_word_dim = n_word_dim
        self.n_char_dim = n_char_dim
        self.n_perspectives = n_perspectives
        self.n_hidden_units = n_hidden_units
        self.n_classes = n_classes
        self.dropout = dropout

        # each word represented with d-dimensional vector with two components
        # TODO character embedding
        self.d = n_word_dim
        # l is the number of perspectives
        self.l = n_perspectives

        self.context_representation_lstm = nn.LSTM(
            input_size=self.d,
            hidden_size=n_hidden_units,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.m_full_forward_W = nn.Parameter(torch.rand(self.l, self.n_hidden_units))
        self.m_full_backward_W = nn.Parameter(torch.rand(self.l, self.n_hidden_units))

        self.aggregation_lstm = nn.LSTM(
            input_size=8*self.l,
            hidden_size=n_hidden_units,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.prediction_layer = nn.Sequential(
            nn.Linear(4*self.n_hidden_units, 2*self.n_hidden_units),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(2*self.n_hidden_units, self.n_classes),
            nn.LogSoftmax(1)
        )

    def matching_strategy_full(self, v1, v2, W):
        """
        :param v1: batch x seq_len x n_hidden
        :param v2: batch x n_hidden
        :param W:  l x n_hidden
        :return: batch x seq_len x l
        """
        l = W.size(0)
        batch_size = v1.size(0)
        seq_len = v1.size(1)

        v1 = v1.unsqueeze(2).expand(-1, -1, l, -1)          # batch x seq_len x l x n_hidden
        W_expanded = W.expand(batch_size, seq_len, -1, -1)  # batch x seq_len x l x n_hidden
        Wv1 = W_expanded.mul(v1)                            # batch x seq_len x l x n_hidden

        v2 = v2.unsqueeze(1).unsqueeze(1).expand(-1, seq_len, l, -1)  # batch x seq_len x l x n_hidden
        Wv2 = W_expanded.mul(v2)

        cos_sim = F.cosine_similarity(Wv1, Wv2, dim=3)
        return cos_sim

    def forward(self, batch):
        # Word Representation Layer
        sent1 = self.word_embedding(batch.sentence_a)
        sent2 = self.word_embedding(batch.sentence_b)

        # Context Representation Layer
        s1_context_out, (s1_context_h, s1_context_c) = self.context_representation_lstm(sent1)
        s2_context_out, (s2_context_h, s2_context_c) = self.context_representation_lstm(sent2)
        s1_context_forward, s1_context_backward = torch.split(s1_context_out, self.n_hidden_units, dim=2)
        s2_context_forward, s2_context_backward = torch.split(s2_context_out, self.n_hidden_units, dim=2)

        # Matching Layer

        # Full matching
        m_full_s1_f = self.matching_strategy_full(s1_context_forward, s2_context_forward[:, -1, :], self.m_full_forward_W)
        m_full_s1_b = self.matching_strategy_full(s1_context_backward, s2_context_backward[:, 0, :], self.m_full_backward_W)
        m_full_s2_f = self.matching_strategy_full(s2_context_forward, s1_context_forward[:, -1, :], self.m_full_forward_W)
        m_full_s2_b = self.matching_strategy_full(s2_context_backward, s1_context_backward[:, 0, :], self.m_full_backward_W)
