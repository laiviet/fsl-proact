import torch
import torch.nn as nn

from .basenet import kl
from .utils import init_weights
from .bert_base import BertEmbedding


class BertMLP(nn.Module):

    def __init__(self, vectors=None, args=None):
        super(BertMLP, self).__init__()
        self.device = args.device
        self.bert = BertEmbedding(args)
        self.dropout = nn.Dropout(args.dropout)
        self.hidden_size = args.hidden_size

        for params in self.bert.parameters():
            params.requires_grad = False

        self.fc1 = nn.Sequential(
            nn.Tanh(),
            nn.Linear(768 * args.bert_layer, args.hidden_size),
        )

        self.mutual = nn.Sequential(
            nn.Dropout(), nn.Sigmoid(), nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh()
        )

    def init_weight(self):
        self.fc1.apply(init_weights)

    def select_anchor(self, emb, anchor_index):
        """

        :param emb: B x L x D
        :param anchor_index: B
        :return:
        """
        B, L, D = emb.shape
        u = torch.tensor([x for x in range(L)]).unsqueeze(0).to(self.device)
        v = anchor_index.view(B, 1)
        mask = (u == v).unsqueeze(dim=2).to(self.device)
        x = torch.masked_select(emb, mask).view(-1, D)
        return x

    def forward(self, inputs):
        embedding = self.bert(inputs)

        # print('| BertMLP > embedding', tuple(embedding.shape))

        full_rep = self.fc1(embedding)

        # print('| BertMLP > full_rep', tuple(full_rep.shape))

        foot_print = inputs['prune_footprint'].unsqueeze(dim=2)

        # print('| BertMLP > foot_print', tuple(foot_print.shape))

        prune_rep = foot_print * full_rep
        # print('| BertMLP > prune_rep', tuple(prune_rep.shape))

        rep = self.select_anchor(self.dropout(full_rep), inputs['anchor_index'])

        # print('| BertMLP > rep', tuple(rep.shape))

        m1 = self.mutual(full_rep)
        m2 = self.mutual(prune_rep)
        mutual_loss = kl(m1, m2)

        return {
            'embedding': rep,
            'mutual_loss': mutual_loss
        }
