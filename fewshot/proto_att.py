import torch
import torch.nn as nn
import torch.nn.functional as F

from fewshot.base import *


class AttPrototypicalNetwork(FSLBaseModel):
    def __init__(self, encoder, args):
        super(AttPrototypicalNetwork, self).__init__(encoder, args)
        self.drop = nn.Dropout()
        # for instance-level attention
        hidden_size = encoder.hidden_size
        self.fc = nn.Linear(hidden_size, hidden_size, bias=True)

    def init_weight(self):
        self.apply(init_weights)

    def __dist__(self, x, y, dim, score=None):
        if score is None:
            return (torch.pow(x - y, 2)).sum(dim)
        else:
            return (torch.pow(x - y, 2) * score).sum(dim)

    def __batch_dist__(self, S, Q, score=None):
        return self.__dist__(S, Q.unsqueeze(2), 3, score)

    def forward(self, batch, setting):
        B, N, K, Q = setting

        encoded = self.encoder(batch)
        embeddings = encoded['embedding']
        D = embeddings.shape[-1]
        embeddings = embeddings.view(B, N, K + Q, D)
        # print('| ProtoHATT embedding', tuple(embeddings.shape))
        support = embeddings[:, :, :K, :].contiguous()  # B x N x K x D
        query = embeddings[:, :, K:, :].contiguous().view(B, N * Q, -1)  # B x NQ x D

        support = support.unsqueeze(1).expand(-1, N * Q, -1, -1, -1)  # (B, NQ, 2*N, K, D)
        # print("ProtoHATT > support", tuple(support.shape))
        # print("ProtoHATT query", tuple(query.shape))

        support_for_att = self.fc(support)
        query_for_att = self.fc(query.unsqueeze(2).unsqueeze(3).expand(-1, -1, N, K, -1))

        ins_att_score = F.softmax(torch.tanh(support_for_att * query_for_att).sum(-1), dim=-1)  # (B, NQ, N, K)
        prototypes = (support * ins_att_score.unsqueeze(4).expand(-1, -1, -1, -1, D)).sum(3)  # (B, NQ, N, D)

        query = query.unsqueeze(dim=2)
        # print('| ProtoHATT query', tuple(query.shape))
        # print('| ProtoHATT prototypes', tuple(prototypes.shape))

        error = query - prototypes  # B x NQ x N x D
        distances = torch.sum(torch.pow(error, 2), dim=3)  # B x NQ x N
        # print('| ProtoHATT distances', tuple(distances.shape))

        return_item = {
            'logit': -distances
        }

        for k, v in encoded.items():
            if k != 'embedding':
                return_item[k] = v
        return return_item
