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

        for i in range(8):
            setattr(self, f'matching_weight_{i}', nn.Parameter(torch.rand(self.l, self.n_hidden_units)))

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

    def matching_strategy_full(self):
        pass

    def forward(self, batch):
        sent1 = self.embedding(batch.sentence_a)
        sent2 = self.embedding(batch.sentence_b)

        s1_context_out, s1_context_h, s1_context_c = self.context_representation_lstm(sent1)
        s2_context_out, s2_context_h, s2_context_c = self.context_representation_lstm(sent2)
