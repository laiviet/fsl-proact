import torch.nn as nn
import torch
import torch.nn.functional as F

from sentence_encoder.basenet import BaseNet, kl
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class GRU(BaseNet):

    def __init__(self, vectors, args):
        super(GRU, self).__init__(vectors, args)
        self.device = args.device
        self.hidden_size = args.hidden_size
        self.window = args.window
        self.max_length = args.max_length
        self.gru = nn.GRU(input_size=self.embedder.hidden_size,
                          hidden_size=args.hidden_size,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)
        self.fc = nn.Sequential(nn.Dropout(),
                                nn.Tanh(),
                                nn.Linear(args.hidden_size * 2, args.hidden_size),
                                nn.Tanh())

        self.mutual = nn.Sequential(
            nn.Dropout(), nn.Sigmoid(), nn.Linear(2 * self.hidden_size, self.hidden_size), nn.Tanh()
        )

    def init_weight(self):
        self.apply(init_weights)

    def initHidden(self, batch_size):
        h0 = torch.zeros(2 * 1, batch_size, self.hidden_size).to(self.device)
        return h0

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
        embedding = self.embedder(inputs['indices'], inputs['dist'], inputs['mask'])
        # print('| GRUEncoder: embedding > ', tuple(embedding.shape))

        B, T, _ = embedding.shape

        # print('| GRUEncoder: anchors > ', tuple(anchors.shape))

        h0 = self.initHidden(B)
        hidden_states, _ = self.gru(embedding, h0)
        rnnRep = self.select_anchor(hidden_states, inputs['anchor_index'])
        pool_x1 = F.max_pool1d(hidden_states.transpose(1, 2), T).squeeze(dim=2)

        # prune_embedding = self.embedder(inputs['prune_indices'], inputs['dist'], inputs['prune_mask'])
        # hidden_states, _ = self.gru(prune_embedding, h0)
        # pool_x2 = F.max_pool1d(hidden_states.transpose(1, 2), T).squeeze(dim=2)

        foot_print = inputs['prune_footprint'].unsqueeze(dim=2)
        # print('| GRUEncoder: foot_print > ', tuple(foot_print.shape))
        # print('| GRUEncoder: hidden_states > ', tuple(hidden_states.shape))

        hidden_states = hidden_states * foot_print

        pool_x2 = F.max_pool1d(hidden_states.transpose(1, 2), T).squeeze(dim=2)

        # print('| GRUEncoder: pool_x1 > ', tuple(pool_x1.shape))
        # print('| GRUEncoder: pool_x2 > ', tuple(pool_x2.shape))

        m1 = self.mutual(pool_x1)
        m2 = self.mutual(pool_x2)

        mutual_loss = kl(m1, m2)

        x = self.fc(rnnRep)

        return {'embedding': x,
                'mutual_loss': mutual_loss
                }
