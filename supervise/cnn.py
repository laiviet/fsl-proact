import torch
import torch.nn as nn
from sentence_encoder.basenet import BaseNet
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def kl(logits1, logits2):
    p1 = torch.softmax(logits1, dim=1)
    p2 = torch.softmax(logits2, dim=1)

    mutual_loss = torch.sum(p1 * ((p1 / p2).log()))
    return mutual_loss


class CNNClassifier(BaseNet):

    def __init__(self, vectors, args):
        super(CNNClassifier, self).__init__(vectors, args)

        # if self['useRelDep']: embedding_input_dim += self['numRelDep']
        modules = []
        for k in self.cnn_kernel_sizes:
            modules.append(nn.Conv2d(1, self.cnn_kernel_number,
                                     (k, self.embedding_input_dim),
                                     padding=(k // 2, 0)
                                     ))

        self.convs = nn.ModuleList(modules)
        self.dim_rep = self.cnn_kernel_number * len(self.cnn_kernel_sizes)

        self.mutual = nn.Sequential(
            nn.Dropout(), nn.Sigmoid(), nn.Linear(self.dim_rep, self.hidden_size), nn.Tanh()
        )

        self.createFcModule(self.dim_rep)

    def init_weight(self):
        super(CNNClassifier, self).init_weight()
        self.convs.apply(init_weights)

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

    def cnn_forward(self, indices, dist, mask):
        inRep = self.embedder(indices, dist, mask)
        # print('| CNN > inRep', tuple(inRep.shape))
        inRep = inRep.unsqueeze(1)  # (B,1,T,D)
        # print('| CNN > inRep', tuple(inRep.shape))

        convRep = [torch.tanh(conv(inRep)).squeeze(3) for conv in self.convs]  # [(B,Co,T), ...]*len(Ks)
        convRep = [x[:, :, :40] for x in convRep]
        # print('| CNN > convRep', tuple(convRep[0].shape))

        pooledRep = [F.max_pool1d(cr, cr.size(2)).squeeze(2) for cr in convRep]  # [(B,Co), ...]*len(Ks)
        # print('| CNN > pooledRep', tuple(pooledRep[0].shape))
        return pooledRep

    def forward(self, inputs):
        inRep = self.embedder(inputs['indices'], inputs['dist'], inputs['mask'])
        # print('| CNN > inRep', tuple(inRep.shape))
        inRep = inRep.unsqueeze(1)  # (B,1,T,D)
        # print('| CNN > inRep', tuple(inRep.shape))

        convRep = [torch.tanh(conv(inRep)).squeeze(3) for conv in self.convs]  # [(B,Co,T), ...]*len(Ks)
        convRep = [x[:, :, :40] for x in convRep]
        # print('| CNN > convRep', tuple(convRep[0].shape))

        full_pooled = [F.max_pool1d(cr, cr.size(2)).squeeze(2) for cr in convRep]  # [(B,Co), ...]*len(Ks)
        concat_full_pooled = torch.cat(full_pooled, dim=-1)

        frep = self.introduceLocalEmbedding(full_pooled, inputs)
        # print('| CNN > frep', tuple(frep.shape))

        logits = self.fc(self.dropout(frep)).view(frep.shape[0], -1)
        # print('| CNN > frep', tuple(frep.shape))

        mask = 1 - inputs['prune_footprint']
        # print('| CNN > mask', tuple(mask.shape))

        mask = mask.bool().unsqueeze(dim=1).expand(-1, self.cnn_kernel_number, -1)
        # print('| CNN > mask', tuple(mask.shape))

        filled_convRep = [torch.masked_fill(x, mask, -1e-12) for x in convRep]
        prune_pooled = [F.max_pool1d(cr, cr.size(2)).squeeze(2) for cr in filled_convRep]
        concat_prune_pooled = torch.cat(prune_pooled, dim=-1)

        # print('| CNN > concat_full_pooled', tuple(concat_full_pooled.shape))
        # print('| CNN > concat_prune_pooled', tuple(concat_prune_pooled.shape))

        m1 = self.mutual(concat_prune_pooled)
        m2 = self.mutual(concat_full_pooled)
        mutual_loss = kl(m1, m2)

        return {'logit': logits,
                'mutual_loss': mutual_loss
                }

    def forwardv1(self, inputs):
        full_rep = self.cnn_forward(inputs['indices'], inputs['dist'], inputs['mask'])
        # print('| CNN > pooledRep', tuple(pooledRep[0].shape))
        prune_rep = self.cnn_forward(inputs['prune_indices'], inputs['dist'], inputs['prune_mask'])
        concat_full_rep = torch.cat(full_rep, dim=-1)
        concat_prune_rep = torch.cat(prune_rep, dim=-1)

        frep = self.introduceLocalEmbedding(full_rep, inputs)
        # print('| CNN > frep', tuple(frep.shape))

        frep = self.fc(self.dropout(frep)).view(frep.shape[0], -1)
        # print('| CNN > frep', tuple(frep.shape))

        # print('| CNN > concat_full_rep', tuple(concat_full_rep.shape))
        # print('| CNN > concat_prune_rep', tuple(concat_prune_rep.shape))

        m1 = self.mutual(concat_full_rep)
        m2 = self.mutual(concat_prune_rep)
        mutual_loss = kl(m1, m2)

        return {'embedding': frep,
                'mutual_loss': mutual_loss
                }
