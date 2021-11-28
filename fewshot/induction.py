# -*- coding: utf-8 -*-
"""
Created on: 2019/5/27 14:29
@Author: zsfeng
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.01)

class NeuralTensorLayer(nn.Module):

    def __init__(self, input_dim, n_slice):
        """

        :param n_class:
        :param input_dim:
        :param output_dim:
        """
        super(NeuralTensorLayer, self).__init__()
        self.slices = nn.ModuleList([nn.Linear(input_dim, input_dim, bias=False) for _ in range(n_slice)])
        self.linear = nn.Linear(n_slice, 1, bias=True)
        self.n_slice = n_slice
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, prototype, query):
        """

        :param prototype: B x N x D
        :param query:   B x NQ x D
        :return:
        """
        # print('| NTL > prototype: ', tuple(prototype.shape))
        # print('| NTL > query: ', tuple(query.shape))

        mid_pro = []
        for i in range(self.n_slice):
            class_slide = self.slices[i](prototype)  # (N,D)
            # print('| NTL > class_slide: ', tuple(class_slide.shape))
            slide_inter = torch.bmm(class_slide, query.transpose(2, 1))  # (N,D) x (DxQ) -> (N,Q)
            # print('| NTL > slide_inter: ', tuple(slide_inter.shape))
            mid_pro.append(slide_inter)
        tensor_bi_product = torch.stack(mid_pro, dim=-1)
        # print('| NTL > tensor_bi_product: ', tuple(tensor_bi_product.shape))
        V = self.relu(torch.transpose(tensor_bi_product, dim0=1, dim1=2))
        # print('| NTL > V: ', tuple(V.shape))

        probs = self.linear(V).squeeze(dim=-1)
        # print('| NTL > probs: ', tuple(probs.shape))

        return probs


def test_neural_tensor_layer():
    B = 4
    N = 5
    K = 6
    Q = 7
    D = 11
    class_vector = torch.zeros(size=(B, N, D))
    query_encoder = torch.zeros(size=(B, Q, D))

    NTL = NeuralTensorLayer(D, 2)
    print(NTL)
    probs = NTL(class_vector, query_encoder)


class Squash(nn.Module):

    def __init__(self):
        super(Squash, self).__init__()

    def forward(self, x):
        """

        :param x: N x D
        :return:
        """
        norm = torch.sum(x * x, dim=-1, keepdim=True)  # (N,1)
        scalar_factor = norm / (1 + norm) / torch.sqrt(norm + 1e-9)  # (N,1)
        vec_squashed = scalar_factor * x  # element-wise  # (N,D)
        return vec_squashed


class DynamicRouting(nn.Module):

    def __init__(self, input_dim=128, hidden_size=128, iteration=3, device='cpu'):
        super(DynamicRouting, self).__init__()
        assert type(iteration) == int, 'iteration must be integer, type(iteration)=' + str(type(iteration))
        assert iteration >= 1, 'Number of iteration has to be positive number, i=' + str(iteration)
        self.iteration = iteration
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.proj = nn.Linear(input_dim, hidden_size)
        self.squash = Squash()
        self.device = device

    def forward(self, support):
        """

        :param support: N x K x D
        :return:
        """
        N, K, D = support.shape
        b_ij = torch.zeros(size=(N, K, 1), dtype=torch.float32).to(self.device)  # N,K,1
        # b_ij = torch.autograd.Variable(b_ij)

        e_ij = self.proj(support)  # (N,K,D)
        e_ij = self.squash(e_ij)

        for i in range(self.iteration):
            d_i = F.softmax(b_ij, dim=1)  # (N,K,1)
            c_i = d_i * e_ij  # (N,K,D)
            c_i = c_i.sum(dim=1)  # N,D
            c_i = self.squash(c_i)  # N,D
            if i < self.iteration - 1:
                delta_b = torch.matmul(e_ij, c_i.unsqueeze(dim=2))  # (N,K,D)x(N,D,1) -> (N,K,1)
                b_ij = b_ij + delta_b
        return c_i

    def project(self, query):
        return self.squash(self.proj(query))


class InductionNetwork(torch.nn.Module):
    def __init__(self, encoder, args):
        super(InductionNetwork, self).__init__()
        self.encoder = encoder
        self.args = args
        D = self.encoder.hidden_size
        self.dynamic_routing = DynamicRouting(D, D, args.induction_iteration, args.device)
        self.neural_tensor_layer = NeuralTensorLayer(D, 100)


    def init_weight(self):
        self.neural_tensor_layer.apply(init_weights)
        self.dynamic_routing.apply(init_weights)


    def forward(self, batch, setting):
        B, N, K, Q = setting

        encoded = self.encoder(batch)
        embeddings = encoded['embedding']
        D = embeddings.shape[-1]
        embeddings = embeddings.view(B, N, K + Q, D)
        # print('| InductionNetwork > embedding', tuple(embeddings.shape))
        support = embeddings[:, :, :K, :].contiguous()  # B x N x K x D
        query = embeddings[:, :, K:, :].contiguous()  # B x N x Q x D

        support = support.view(-1, K, D)  # (BN, K, D)

        # print('| InductionNetwork > support', tuple(support.shape))
        # print('| InductionNetwork > query', tuple(query.shape))

        prototypes = self.dynamic_routing(support).view(B, N, -1)

        query = query.view(B, -1, D)  # (BN, Q, D)
        # query = self.dynamic_routing.project(query)
        logits = self.neural_tensor_layer(prototypes, query)
        # print('| InductionNetwork > logits', tuple(logits.shape))

        return_item = {
            'logit': logits
        }

        for k, v in encoded.items():
            if k != 'embedding':
                return_item[k] = v
        return return_item


if __name__ == '__main__':
    test_neural_tensor_layer()
