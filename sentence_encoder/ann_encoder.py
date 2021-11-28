import torch
import torch.nn as nn

from .basenet import *


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class ANN(nn.Module):

    def __init__(self, vectors, args):
        super(ANN, self).__init__()
        self.device = args.device
        word_tensors = torch.from_numpy(vectors).float()
        embedding_size = word_tensors.shape[1]
        hidden_size = args.hidden_size
        entity_size = 50

        self.word_embedder = nn.Embedding(word_tensors.shape[0], word_tensors.shape[1]).cuda()
        self.word_embedder.weight.data.copy_(word_tensors)
        self.word_embedder.weight.requires_grad = args.tune_embedding

        self.entity_embedder = nn.Embedding(100, entity_size)

        self.f_w = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.Tanh()
        )

        self.f_e = nn.Sequential(
            nn.Linear(embedding_size, entity_size),
            nn.Tanh()
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(embedding_size * 2 + entity_size, hidden_size),
            nn.Tanh(),
        )

        self.mse = nn.MSELoss()

    def init_weight(self):
        self.apply(init_weights)

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

    def attention(self, anchor, embedding):
        """

        :param anchor: B x D
        :param embedding: B x L x D
        :return:
        """

        anchor = anchor.unsqueeze(dim=-1)  # B,D,1
        scores = torch.bmm(embedding, anchor)  # B,L,D x B,D,1 -> B,L,1
        scores = torch.exp(scores - scores.max())  # B,L,1 Using max to avoid exponential instability
        sum_scores = scores.sum(dim=1, keepdim=True)  # B,1,1
        alpha = scores / sum_scores  # B,L,1
        return alpha

    def forward(self, inputs):
        mask = inputs['mask'].unsqueeze(dim=-1)  # B,L,1
        entity_attention = inputs['entity_attention']
        w = self.word_embedder(inputs['indices'])  # B,L,D
        w = w * mask
        w_bar = self.f_w(w)  # B,L,D
        anchor_w_bar = self.select_anchor(w_bar, inputs['anchor_index'])  # B,D

        anchor_w = self.select_anchor(w, inputs['anchor_index'])  # B,D
        anchor_e = self.f_e(anchor_w)  # B,D
        e = self.entity_embedder(inputs['entity_indices'])  # B,L,D
        e = e * mask

        alpha_w = self.attention(anchor_w_bar, w_bar)  # B,L,1
        alpha_e = self.attention(anchor_e, e)  # B,L,1

        alpha = alpha_w + alpha_e  # B,L,1

        # print('|ANN > alpha', tuple(alpha.shape))
        # print('|ANN > entity_attention', tuple(entity_attention.shape))

        c_w = torch.sum(alpha * w, dim=1)  # B,L,1 x B,L,D -> B,L,D -> B,D
        c_e = torch.sum(alpha * e, dim=1)  # B,L,1 x B,L,D -> B,L,D -> B,D

        final_rep = torch.cat([c_w, c_e, anchor_w], dim=-1)  # B,3D
        rep = self.fc(final_rep)  # B,D

        attention_loss = self.mse(alpha.squeeze(dim=2), entity_attention)

        return {
            'embedding': rep,
            'attention_loss': attention_loss
        }
