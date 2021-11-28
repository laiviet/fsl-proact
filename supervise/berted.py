# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
from sentence_encoder.bert_base import BertEmbedding

INFINITY_NUMBER = 1e12


class BertEDClassifier(nn.Module):
    def __init__(self, vectors, args):
        super(BertEDClassifier, self).__init__()
        # self.squeeze_embedding = SqueezeEmbedding()
        self.device = args.device
        self.hidden_size = args.hidden_size
        self.bert = BertEmbedding(args)
        self.dropout = nn.Dropout(args.dropout)
        self.bert_layer = args.bert_layer
        self.fc = nn.Linear(768 * self.bert_layer, args.n_class)

    def forward(self, inputs):
        """
        bert_length: L
        original_length: T
        """
        anchor_index = inputs['anchor_index']

        embeddings = self.bert(inputs)
        B, T, D = embeddings.shape

        mask = torch.LongTensor([i for i in range(T)]).to(self.device).repeat(B, 1)  # B x T
        # print('| BertED > mask', tuple(mask.shape))

        mask = mask == anchor_index.unsqueeze(1)
        mask = mask.unsqueeze(dim=2).expand(-1, -1, 768 * self.bert_layer)
        # print('| BertED > mask', tuple(mask.shape))

        anchor_rep = torch.masked_select(embeddings, mask).view(B, -1)
        # print('| BertED > anchor_rep', tuple(anchor_rep.shape))

        anchor_rep = self.dropout(anchor_rep)
        logits = self.fc(anchor_rep)
        # print('| BertED > logits', tuple(logits.shape))

        return {
            'logit': logits
        }
