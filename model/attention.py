'''
    pytorch implementation of our RMN model
'''

import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.allennlp_beamsearch import BeamSearch
from torch.nn.utils.weight_norm import weight_norm
import math
import numpy as np

# ------------------------------------------------------
# ------------ Soft Attention Mechanism ----------------
# ------------------------------------------------------
class SoftDotAttention(nn.Module):
    def __init__(self, dim_ctx, dim_h):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim_h, dim_ctx, bias=False)
        self.sm = nn.Softmax(dim=1)

    def forward(self, context, h, mask=None):
        '''Propagate h through the network.
        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1
        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        # if mask is not None:
        #     # -Inf masking prior to the softmax
        #     attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len
        weighted_ctx = torch.bmm(attn3, context) # batch x dim
        return weighted_ctx, attn


class SoftAttention(nn.Module):
    def __init__(self, feat_size, hidden_size, att_size, dropout=0.1):
        super(SoftAttention, self).__init__()
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.wh = nn.Linear(hidden_size, att_size)
        self.wv = nn.Linear(feat_size, att_size)
        self.wa = nn.Linear(att_size, 1, bias=False)
        # self.drop=nn.Dropout(p=dropout)
    def forward(self, feats, key):
        '''
        :param feats: (batch_size, feat_num, feat_size)
        :param key: (batch_size, hidden_size)
        :return: att_feats: (batch_size, feat_size)
                 alpha: (batch_size, feat_num)
        '''
        v = self.wv(feats)
        inputs = self.wh(key).unsqueeze(1).expand_as(v) + v
        alpha = F.softmax(self.wa(torch.tanh(inputs)).squeeze(-1), dim=1)
        att_feats = torch.bmm(alpha.unsqueeze(1), feats).squeeze(1)
        return att_feats, alpha


# ------------------------------------------------------
# ------------ Self Attention Mechanism --------------
# ------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, feat_size, hidden_size, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.q_liner = nn.Linear(feat_size, hidden_size)
        self.k_liner = nn.Linear(feat_size, hidden_size)
        self.v_liner = nn.Linear(feat_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, mask=None, dropout=None):
        q, k, v = self.q_liner(input), self.k_liner(input), self.v_liner(input)
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) \
                 / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, v), p_attn


if __name__ == '__main__':
    feat = torch.randn(5,26,512)
    key = torch.randn(5, 26,512)
    att = GumbelTopkAttention(512,512,512)
    att_feat, alpha = att(feat,key)
    print(att_feat)
    # feat = torch.randn(5, 26, 512)
    # att = SelfAttention(512, 512)
    # weight = att(feat)
    # print(weight[0].shape)









