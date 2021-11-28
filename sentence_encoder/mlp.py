import torch
import torch.nn as nn

from .basenet import kl
from .utils import init_weights
from .bert_base import BertEmbedding


class MLP(nn.Module):

    def __init__(self, vectors=None, args=None):
        super(MLP, self).__init__()
        self.device = args.device
        self.dropout = nn.Dropout(args.dropout)
        self.bert_layer = args.bert_layer

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(768 * args.bert_layer, 512),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(512, args.hidden_size),
        )

    def init_weight(self):
        self.fc1.apply(init_weights)

    def forward(self, inputs):
        embedding = inputs['emb'][:, -self.bert_layer:, :]
        # print('| MLP > embedding', tuple(embedding.shape))
        embedding = embedding.view(embedding.shape[0], -1)
        # print('| MLP > embedding', tuple(embedding.shape))

        full_rep = self.fc1(embedding)
        # print('| MLP > full_rep', tuple(full_rep.shape))

        # exit(0)

        rep = self.dropout(full_rep)
        return {
            'embedding': rep,
        }
