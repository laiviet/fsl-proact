import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .basenet import kl
from .utils import init_weights
from .bert_base import BertEmbedding


class BertCNN(nn.Module):
    def __init__(self, vectors, args):
        super(BertCNN, self).__init__()
        # self.squeeze_embedding = SqueezeEmbedding()
        self.device = args.device
        self.bert = BertEmbedding(args)

        self.dropout = nn.Dropout(args.dropout)
        self.hidden_size = args.hidden_size
        self.bert_layer = args.bert_layer

        self.cnn_kernel_number = 512

        modules = []
        for k in args.cnn_kernel_sizes:
            modules.append(nn.Conv2d(1, self.cnn_kernel_number,
                                     (k, 768 * self.bert_layer),
                                     padding=(k // 2, 0)))

        self.convs = nn.ModuleList(modules)
        self.dim_rep = self.cnn_kernel_number * len(args.cnn_kernel_sizes)

        self.mutual = nn.Sequential(
            nn.Dropout(), nn.Sigmoid(), nn.Linear(self.dim_rep, self.hidden_size), nn.Tanh()
        )

        self.createFcModule(self.dim_rep)

        for params in self.bert.parameters():
            params.requires_grad = args.bert_update

    def init_weight(self):
        self.convs.apply(init_weights)

    def createFcModule(self, dim_rep):

        dim_rep = dim_rep + self.bert_layer * 768

        rep_hids = [dim_rep, self.hidden_size, self.hidden_size]

        ofcs = OrderedDict()
        for i, (ri, ro) in enumerate(zip(rep_hids, rep_hids[1:])):
            ofcs['finalRep_' + str(i)] = nn.Linear(ri, ro)
            ofcs['finalNL_' + str(i)] = nn.Tanh()
        self.fc = nn.Sequential(ofcs)

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

    def cnn_forward(self, embedding):

        inRep = embedding.unsqueeze(1)  # (B,1,T,D)
        # print('| CNN > inRep', tuple(inRep.shape))

        convRep = [torch.tanh(conv(inRep)).squeeze(3) for conv in self.convs]  # [(B,Co,T), ...]*len(Ks)
        convRep = [x[:, :, :40] for x in convRep]
        # print('| CNN > convRep', tuple(convRep[0].shape))

        pooledRep = [F.max_pool1d(cr, cr.size(2)).squeeze(2) for cr in convRep]  # [(B,Co), ...]*len(Ks)
        # print('| CNN > pooledRep', tuple(pooledRep[0].shape))
        return pooledRep

    def forward(self, inputs):
        embedding = self.bert(inputs)
        # print('| BertCNN > embedding', tuple(embedding.shape))

        anchor = self.select_anchor(embedding, inputs['anchor_index'])
        # print('| BertCNN > anchor', tuple(anchor.shape))

        inRep = embedding.unsqueeze(1)  # (B,1,T,D)
        # print('| CNN > inRep', tuple(inRep.shape))

        convRep = [torch.tanh(conv(inRep)).squeeze(3) for conv in self.convs]  # [(B,Co,T), ...]*len(Ks)
        convRep = [x[:, :, :40] for x in convRep]
        # print('| CNN > convRep', tuple(convRep[0].shape))

        full_pooled = [F.max_pool1d(cr, cr.size(2)).squeeze(2) for cr in convRep]  # [(B,Co), ...]*len(Ks)
        concat_full_pooled = torch.cat(full_pooled, dim=-1)

        mask = 1 - inputs['prune_footprint']
        # print('| CNN > mask', tuple(mask.shape))

        mask = mask.bool().unsqueeze(dim=1).expand(-1, self.cnn_kernel_number, -1)
        # print('| CNN > mask', tuple(mask.shape))

        filled_convRep = [torch.masked_fill(x, mask, -1e-12) for x in convRep]
        prune_pooled = [F.max_pool1d(cr, cr.size(2)).squeeze(2) for cr in filled_convRep]
        concat_prune_pooled = torch.cat(prune_pooled, dim=-1)

        m1 = self.mutual(concat_full_pooled)
        m2 = self.mutual(concat_prune_pooled)
        mutual_loss = kl(m1, m2)

        rep = torch.cat(full_pooled + [anchor], dim=-1)

        frep = self.fc(self.dropout(rep)).view(rep.shape[0], -1)

        return {'embedding': frep,
                'mutual_loss': mutual_loss
                }

    def forwardv1(self, inputs):
        embedding = self.bert(inputs)
        # print('| BertCNN > embedding', tuple(embedding.shape))
        anchor = self.select_anchor(embedding, inputs['anchor_index'])
        # print('| BertCNN > anchor', tuple(anchor.shape))

        full_rep = self.cnn_forward(embedding)
        foot_print = inputs['prune_footprint'].unsqueeze(dim=2)

        prune_embedding = embedding * foot_print

        # print('| BertCNN > foot_print', tuple(foot_print.shape))
        # print('| BertCNN > full_rep[0]', tuple(full_rep[0].shape))
        prune_rep = self.cnn_forward(prune_embedding)

        m1 = self.mutual(torch.cat(full_rep, dim=-1))
        m2 = self.mutual(torch.cat(prune_rep, dim=-1))
        mutual_loss = kl(m1, m2)

        rep = torch.cat(full_rep + [anchor], dim=-1)

        frep = self.fc(self.dropout(rep)).view(rep.shape[0], -1)

        return {'embedding': frep,
                'mutual_loss': mutual_loss
                }
