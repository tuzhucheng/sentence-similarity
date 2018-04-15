"""
This implementation is partially inspired by https://github.com/galsang/BIMPM-pytorch/blob/master/model/BIMPM.py
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

        self.m_full_forward_W = nn.Parameter(torch.rand(self.l, self.n_hidden_units))       # W^1 in paper
        self.m_full_backward_W = nn.Parameter(torch.rand(self.l, self.n_hidden_units))      # W^2 in paper
        self.m_maxpool_forward_W = nn.Parameter(torch.rand(self.l, self.n_hidden_units))    # W^3 in paper
        self.m_maxpool_backward_W = nn.Parameter(torch.rand(self.l, self.n_hidden_units))   # W^4 in paper
        self.m_attn_forward_W = nn.Parameter(torch.rand(self.l, self.n_hidden_units))       # W^5 in paper
        self.m_attn_backward_W = nn.Parameter(torch.rand(self.l, self.n_hidden_units))      # W^6 in paper

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
        :param v2: batch x n_hidden (FULL) or batch x seq_len x n_hidden (ATTENTIVE)
        :param W:  l x n_hidden
        :return: batch x seq_len x l
        """
        l = W.size(0)
        batch_size = v1.size(0)
        seq_len = v1.size(1)

        v1 = v1.unsqueeze(2).expand(-1, -1, l, -1)          # batch x seq_len x l x n_hidden
        W_expanded = W.expand(batch_size, seq_len, -1, -1)  # batch x seq_len x l x n_hidden
        Wv1 = W_expanded.mul(v1)                            # batch x seq_len x l x n_hidden

        if len(v2.size()) == 2:
            v2 = v2.unsqueeze(1).unsqueeze(1).expand(-1, seq_len, l, -1)  # batch x seq_len x l x n_hidden
        elif len(v2.size()) == 3:
            v2 = v2.unsqueeze(2).expand(-1, -1, l, -1)  # batch x seq_len x l x n_hidden
        else:
            raise ValueError(f'Invalid v2 tensor size {v2.size()}')
        Wv2 = W_expanded.mul(v2)

        cos_sim = F.cosine_similarity(Wv1, Wv2, dim=3)
        return cos_sim

    def matching_strategy_pairwise(self, v1, v2, W):
        """
        Used as a subroutine for (2) Maxpooling-Matching
        :param v1: batch x seq_len_1 x n_hidden
        :param v2: batch x seq_len_2 x n_hidden
        :param W: l x n_hidden
        :return: batch x seq_len_1 x seq_len_2 x l
        """
        l = W.size(0)
        batch_size = v1.size(0)

        v1_expanded = v1.unsqueeze(1).expand(-1, l, -1, -1)                 # batch x l x seq_len_1 x n_hidden
        W1_expanded = W.unsqueeze(1).expand(batch_size, -1, v1.size(1), -1) # batch x l x seq_len_1 x n_hidden
        Wv1 = W1_expanded.mul(v1_expanded)                                  # batch x l x seq_len_1 x n_hidden

        v2_expanded = v2.unsqueeze(1).expand(-1, l, -1, -1)                 # batch x l x seq_len_2 x n_hidden
        W2_expanded = W.unsqueeze(1).expand(batch_size, -1, v2.size(1), -1) # batch x l x seq_len_2 x n_hidden
        Wv2 = W2_expanded.mul(v2_expanded)                                  # batch x l x seq_len_2 x n_hidden

        dot = torch.matmul(Wv1, Wv2.transpose(3,2))
        v1_norm = v1_expanded.norm(p=2, dim=3, keepdim=True)
        v2_norm = v2_expanded.norm(p=2, dim=3, keepdim=True)
        norm_product = torch.matmul(v1_norm, v2_norm.transpose(3,2))

        cosine_matrix = dot / norm_product
        cosine_matrix = cosine_matrix.permute(0, 2, 3, 1)

        return cosine_matrix

    def matching_strategy_attention(self, v1, v2):
        """
        Used as a subroutine for (3) Attentive-Matching
        :param v1: batch x seq_len_1 x n_hidden
        :param v2: batch x seq_len_2 x n_hidden
        :return: batch x seq_len_1 x seq_len_2
        """
        dot = torch.bmm(v1, v2.transpose(2, 1))
        v1_norm = v1.norm(p=2, dim=2, keepdim=True)
        v2_norm = v2.norm(p=2, dim=2, keepdim=True)
        norm_product = torch.bmm(v1_norm, v2_norm.transpose(2, 1))

        return dot / norm_product

    def forward(self, batch):
        # Word Representation Layer
        sent1 = self.word_embedding(batch.sentence_a)
        sent2 = self.word_embedding(batch.sentence_b)

        # Context Representation Layer
        s1_context_out, (s1_context_h, s1_context_c) = self.context_representation_lstm(sent1)
        s2_context_out, (s2_context_h, s2_context_c) = self.context_representation_lstm(sent2)
        # s1_context_forward and backward are dimensions batch x seq_len_1 x n_hidden
        s1_context_forward, s1_context_backward = torch.split(s1_context_out, self.n_hidden_units, dim=2)
        # s2_context_forward and backward are dimensions batch x seq_len_2 x n_hidden
        s2_context_forward, s2_context_backward = torch.split(s2_context_out, self.n_hidden_units, dim=2)

        # Matching Layer

        # (1) Full matching
        m_full_s1_f = self.matching_strategy_full(s1_context_forward, s2_context_forward[:, -1, :], self.m_full_forward_W)
        m_full_s1_b = self.matching_strategy_full(s1_context_backward, s2_context_backward[:, 0, :], self.m_full_backward_W)
        m_full_s2_f = self.matching_strategy_full(s2_context_forward, s1_context_forward[:, -1, :], self.m_full_forward_W)
        m_full_s2_b = self.matching_strategy_full(s2_context_backward, s1_context_backward[:, 0, :], self.m_full_backward_W)

        # (2) Maxpooling-Matching
        m_pair_f = self.matching_strategy_pairwise(s1_context_forward, s2_context_backward, self.m_maxpool_forward_W)
        m_pair_b = self.matching_strategy_pairwise(s1_context_backward, s2_context_backward, self.m_maxpool_backward_W)

        m_maxpool_s1_f, _ = m_pair_f.max(dim=2)
        m_maxpool_s1_b, _ = m_pair_b.max(dim=2)
        m_maxpool_s2_f, _ = m_pair_f.max(dim=1)
        m_maxpool_s3_b, _ = m_pair_b.max(dim=1)

        # (3) Attentive-Matching
        # cosine_f and cosine_b are batch x seq_len_1 x seq_len_2
        cosine_f = self.matching_strategy_attention(s1_context_forward, s2_context_forward)
        cosine_b = self.matching_strategy_attention(s1_context_backward, s2_context_backward)

        # attn_s1_f and others are batch x seq_len_1 x seq_len_2 x n_hidden
        attn_s1_f = s1_context_forward.unsqueeze(2) * cosine_f.unsqueeze(3)
        attn_s1_b = s1_context_forward.unsqueeze(2) * cosine_b.unsqueeze(3)
        attn_s2_f = s2_context_forward.unsqueeze(1) * cosine_f.unsqueeze(3)
        attn_s2_b = s2_context_forward.unsqueeze(1) * cosine_b.unsqueeze(3)

        attn_mean_vec_s2_f = attn_s1_f.sum(dim=1) / cosine_f.sum(1, keepdim=True).transpose(2, 1)  # batch x seq_len_2 x hidden
        attn_mean_vec_s2_b = attn_s1_b.sum(dim=1) / cosine_b.sum(1, keepdim=True).transpose(2, 1)  # batch x seq_len_2 x hidden
        attn_mean_vec_s1_f = attn_s2_f.sum(dim=2) / cosine_f.sum(2, keepdim=True)                  # batch x seq_len_1 x hidden
        attn_mean_vec_s1_b = attn_s2_b.sum(dim=2) / cosine_b.sum(2, keepdim=True)                  # batch x seq_len_1 x hidden

        m_attn_s1_f = self.matching_strategy_full(s1_context_forward, attn_mean_vec_s1_f, self.m_attn_forward_W)
        m_attn_s1_b = self.matching_strategy_full(s1_context_backward, attn_mean_vec_s1_b, self.m_attn_forward_W)
        m_attn_s2_f = self.matching_strategy_full(s2_context_forward, attn_mean_vec_s2_f, self.m_attn_forward_W)
        m_attn_s2_b = self.matching_strategy_full(s2_context_backward, attn_mean_vec_s2_b, self.m_attn_forward_W)
