import torch
import torch.nn as nn

from fewshot.base import *


class MahalanobisDistance(nn.Module):

    def __init__(self, hidden_size=128):
        super(MahalanobisDistance, self).__init__()
        self.m = torch.ones(size=(1, hidden_size), requires_grad=True)

    def forward(self, x, y):
        raise NotImplemented


def test_mahalanobis():
    torch.random.manual_seed(1)

    B = 4
    D = 5
    x = torch.rand(size=(B, D))
    y = torch.rand(size=(B, D))
    print(x)
    print(y)

    A = torch.eye(D)

    err = x - y
    sq = torch.sum(err * err, dim=1)
    sqrt = torch.sqrt(sq)
    print(sqrt)

    maha = MahalanobisDistance(5, cov=False)
    print(maha(x, y))


if __name__ == '__main__':
    test_mahalanobis()


class MahalanobisNetwork(FSLBaseModel):
    def __init__(self, encoder, args):
        super(MahalanobisNetwork, self).__init__(encoder, args)
        self.m = torch.ones(size=(1, 1, args.hidden_size), requires_grad=True)

    def mahalanobis_distance(self, x, y):
        d = x - y
        square = d * d
        square = square * self.m
        sqrt = torch.sqrt(torch.sum(square, dim=1))
        return -sqrt

    def forward(self, batch, setting):
        B, N, K, Q = setting

        encoded = self.encoder(batch)
        embeddings = encoded['embedding']
        D = embeddings.shape[-1]
        embeddings = embeddings.view(B, N, K + Q, D)
        # print('| Proto > embedding', tuple(embeddings.shape))
        support = embeddings[:, :, :K, :].contiguous()  # B x N x K x D
        query = embeddings[:, :, K:, :].contiguous()  # B x N x Q x D

        # print('| Proto > support', tuple(support.shape))
        # print('| Proto > query', tuple(query.shape))

        prototypes = support.mean(dim=2)  # ->  B x N x D
        prototypes = prototypes.unsqueeze(dim=1)  # B x 1 x N x D
        # print('| Proto > prototypes', tuple(prototypes.shape))

        query = query.view(B, N * Q, D)  # ->  B x NQ x D
        # print('| Proto > query', tuple(query.shape))

        query = query.unsqueeze(dim=2)  # ->  B x NQ x 1 x D
        # print('| Proto > query', tuple(query.shape))

        error = query - prototypes  # B x NQ x N x D
        logits = -torch.sum(torch.pow(error, 2), dim=3)  # B x NQ x N

        # print(torch.mean(logits))

        return_item = {
            'logit': logits,
            # 'global_mutual_loss': self.global_mutual_loss(encoded['pool1'], encoded['pool2'], setting),
            'openset_loss': self.openset_loss(embeddings, setting)
        }

        for k, v in encoded.items():
            if k != 'embedding':
                return_item[k] = v
        return return_item
