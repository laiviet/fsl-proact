import torch
import torch.nn as nn
import math

from fewshot.base import *


class ProtoHead(nn.Module):

    def __init__(self):
        super(ProtoHead, self).__init__()

    def euclidean(self, s1, s2):
        return torch.sum(torch.pow(s1 - s2, 2))

    def cosine(self, s1, s2):
        return torch.sum(torch.nn.functional.cosine_similarity(s1, s2, -1))

    def intra_loss(self, support):
        B, N, K, D = support.shape

        support = support.view(-1, K, D)  # BN x K x D
        s1 = support.unsqueeze(dim=1).expand(-1, K, -1, -1)  # BN x K x K x D
        s2 = support.unsqueeze(dim=2).expand(-1, -1, K, -1)  # BN x K x K x D
        return self.euclidean(s1, s2)

    def inter_loss(self, prototypes):
        B, N, D = prototypes.shape
        s1 = prototypes.unsqueeze(dim=1).expand(-1, N, -1, -1)  # B x N x N x D
        s2 = prototypes.unsqueeze(dim=2).expand(-1, -1, N, -1)  # B x N x N x D
        return self.cosine(s1, s2)

    def forward(self, support, query):
        """

        :param support: (B,N,K,D)
        :param query: (B,N,Q,D)
        :return:
        """
        B, N, K, D = support.shape
        Q = query.shape[2]
        prototypes = support.mean(dim=2)  # ->  B x N x D
        unsqueeze_prototypes = prototypes.unsqueeze(dim=1)  # B x 1 x N x D
        # print('| Proto > prototypes', tuple(prototypes.shape))

        query = query.view(B, N * Q, -1)  # ->  B x NQ x D
        # print('| Proto > query', tuple(query.shape))

        query = query.unsqueeze(dim=2)  # ->  B x NQ x 1 x D
        # print('| Proto > query', tuple(query.shape))

        error = query - unsqueeze_prototypes  # B x NQ x N x D
        logits = -torch.sum(torch.pow(error, 2), dim=3)  # B x NQ x N
        # print('| Proto > logits', tuple(logits.shape))

        return logits


class MELRPlus(FSLBaseModel):
    def __init__(self, encoder, args):
        super(MELRPlus, self).__init__(encoder, args)
        self.dropout = nn.Dropout(args.dropout)
        self.head = ProtoHead()
        self.kl = nn.KLDivLoss()

    def ceam(self, query, key, value):
        """

        :param query: B x N x Q x D
        :param key: B x N x K x D
        :param value: B x N x K x D
        :return:
        """
        D = query.shape[-1]

        B, N, Q, D = query.shape
        _, _, K, _ = key.shape

        query = query.contiguous().view(B, N * Q, D)
        key = key.contiguous().view(B, N * K, D)
        value = value.contiguous().view(B, N * K, D)
        # print('query', tuple(query.shape))
        # print('key', tuple(key.shape))
        # print('value', tuple(value.shape))
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(D)
        # print('scores', tuple(scores.shape))

        p_attn = F.softmax(scores, dim=-1)
        # print('p_attn', tuple(p_attn.shape))

        fx = torch.matmul(p_attn, value)

        # print('fx', tuple(fx.shape))
        # print(40 * '-')
        xfx = query + fx

        return xfx.view(B, N, Q, D).contiguous()

    def melr(self, s1, q1, s2, q2):
        """

        :param s1:
        :param q1:
        :param s2:
        :param q2:
        :return:
        """
        B, N, K1, D = s1.shape
        _, _, Q1, _ = q1.shape
        _, _, K2, _ = s2.shape
        _, _, Q2, _ = q2.shape

        Q = Q1 + Q2

        # print('Q=', Q)
        #
        # print('MELRNetwork > s1', tuple(s1.shape))
        # print('MELRNetwork > q1', tuple(q1.shape))
        # print('MELRNetwork > s2', tuple(s2.shape))
        # print('MELRNetwork > q2', tuple(q2.shape))
        f1 = torch.cat([s1, q1], dim=2)
        f2 = torch.cat([s2, q2], dim=2)

        # print('MELRNetwork > f1', tuple(f1.shape))
        # print('MELRNetwork > f2', tuple(f2.shape))

        f1_hat = self.ceam(f1, s2, s2)
        f2_hat = self.ceam(f2, s1, s1)

        # print('MELRNetwork > f1_hat', tuple(f1_hat.shape))
        # print('MELRNetwork > f2_hat', tuple(f2_hat.shape))

        s1_hat, q1_hat = f1_hat[:, :, :K1, :], f1_hat[:, :, K1:, :]
        s2_hat, q2_hat = f2_hat[:, :, :K2, :], f2_hat[:, :, K2:, :]

        # print('MELRNetwork > s1_hat', tuple(s1_hat.shape))
        # print('MELRNetwork > s2_hat', tuple(s2_hat.shape))
        # print('MELRNetwork > q1_hat', tuple(q1_hat.shape))
        # print('MELRNetwork > q2_hat', tuple(q2_hat.shape))

        q_hat = torch.cat([q1, q2], dim=2)
        # q_hat = torch.cat([q1_hat, q2_hat], dim=2)
        # print('MELRNetwork > q_hat', tuple(q_hat.shape))

        logits_cerc_1 = self.head(s1_hat, q_hat).view(B * N, Q, N)
        logits_cerc_2 = self.head(s2_hat, q_hat).view(B * N, Q, N)

        # print('MELRNetwork > logits_cerc_1', tuple(logits_cerc_1.shape))
        # print('MELRNetwork > logits_cerc_2', tuple(logits_cerc_2.shape))

        logit1 = logits_cerc_1[:, :Q1, :]
        logit2 = logits_cerc_2[:, Q1:, :]

        logits = torch.cat([logit1, logit2], dim=1)

        # print('MELRNetwork > logit1', tuple(logit1.shape))
        # print('MELRNetwork > logit2', tuple(logit2.shape))
        # print('MELRNetwork > logits', tuple(logits.shape))

        # epsilon=1e-12 for numeric stability, as KL divergence calculate log
        sm1 = logits_cerc_1.softmax(dim=-1).view(-1, N) + 1e-12
        sm2 = logits_cerc_2.softmax(dim=-1).view(-1, N) + 1e-12

        # print(sm2.sum(-1))

        cerc_loss = self.kl(sm1.log(), sm2)

        pack = {
            'logit': logits,
            'cerc_loss': cerc_loss
        }

        return pack

    def forward(self, batch, setting, training=True):
        # oK, oQ are the original setting
        B, N, oK, oQ = setting

        # K, Q are the sampled setting, in training, K=2xoK
        if training:
            K = oK * 2
            Q = oQ * 2
        else:
            K, Q = oK, oQ
            sK = K // 2
            sQ = Q // 2

            # print('sK, sQ', sK, sQ)

        # for k, v in batch.items():
        #     print(k, v.shape)
        encoded = self.encoder(batch)
        embeddings = encoded['embedding']
        # print('embeddings', embeddings[0][:10])

        D = embeddings.shape[-1]
        embeddings = embeddings.view(B, N, -1, D)
        # print('| Proto > embedding', tuple(embeddings.shape))
        support = embeddings[:, :, :K, :].contiguous()  # B x N x K x D
        query = embeddings[:, :, K:, :].contiguous()  # B x N x Q x D

        # print('support', support[0][0][0][:10])

        # print('MELRNetwork > support', tuple(support.shape))
        # print('MELRNetwork > query', tuple(query.shape))
        # print('MELRNetwork > setting', setting)

        if training:
            s1, s2 = support[:, :, :oK, :], support[:, :, oK:, :]
            q1, q2 = query[:, :, :oQ, :], query[:, :, oQ:, :]
        else:
            s1, s2 = support[:, :, :sK, :], support[:, :, sK:, :]
            q1, q2 = query[:, :, :sQ, :], query[:, :, sQ:, :]
        pack = self.melr(s1, q1, s2, q2)

        return pack
