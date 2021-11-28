import torch
import torch.nn as nn

from torch.nn import init
from fewshot.base import *


class AttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.Wr = nn.Linear(input_size, hidden_size)
        init.xavier_normal_(self.Wr.state_dict()['weight'])
        self.Ur = nn.Linear(hidden_size, hidden_size)
        init.xavier_normal_(self.Ur.state_dict()['weight'])
        self.W = nn.Linear(input_size, hidden_size)
        init.xavier_normal_(self.W.state_dict()['weight'])
        self.U = nn.Linear(hidden_size, hidden_size)
        init.xavier_normal_(self.U.state_dict()['weight'])
        self.dropout = nn.Dropout(0.2)

    def forward(self, fact, C, g):
        '''
        fact.size() -> (#batch, #hidden = #embedding)
        c.size() -> (#hidden, ) -> (#batch, #hidden = #embedding)
        r.size() -> (#batch, #hidden = #embedding)
        h_tilda.size() -> (#batch, #hidden = #embedding)
        g.size() -> (#batch, )
        '''

        # print('AttentionGRUCell > fact > ', tuple(fact.shape))
        # print('AttentionGRUCell > C > ', tuple(C.shape))
        # print('AttentionGRUCell > g > ', tuple(g.shape))

        fact = self.dropout(fact)

        r = torch.sigmoid(self.Wr(fact) + self.Ur(C))
        h_tilda = torch.tanh(self.W(fact) + r * self.U(C))
        # print('AttentionGRUCell > h_tilda > ', tuple(h_tilda.shape))
        # print('AttentionGRUCell > g > ', tuple(g.shape))

        g = g.unsqueeze(2)  # B, Q, 1
        h = g * h_tilda + (1 - g) * C
        # print('AttentionGRUCell > h > ', tuple(h.shape))
        return h


class AttentionGRU(nn.Module):
    def __init__(self, input_size, hidden_size, device='cpu'):
        super(AttentionGRU, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.AGRUCell = AttentionGRUCell(input_size, hidden_size)

    def forward(self, facts, G):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        fact.size() -> (#batch, #hidden = #embedding)
        G.size() -> (#batch, #sentence)
        g.size() -> (#batch, )
        C.size() -> (#batch, #hidden)
        '''
        B, F, D = facts.size()
        _, Q, _ = G.shape
        # print('AttentionGRU > facts > ', tuple(facts.shape))
        # print('AttentionGRU > G > ', tuple(G.shape))
        C = torch.zeros(self.hidden_size).to(self.device)
        for sid in range(F):
            fact = facts[:, sid:sid + 1, :]
            g = G[:, :, sid]
            if sid == 0:
                C = torch.zeros(size=(B, Q, D)).to(self.device)
            C = self.AGRUCell(fact, C, g)
        return C


class EpisodicMemory(nn.Module):
    def __init__(self, hidden_size, device='cpu'):
        super(EpisodicMemory, self).__init__()
        self.device = device
        self.AGRU = AttentionGRU(hidden_size, hidden_size, device)
        self.z1 = nn.Linear(4 * hidden_size, hidden_size)
        self.z2 = nn.Linear(hidden_size, 1)
        self.next_mem = nn.Linear(3 * hidden_size, hidden_size)
        init.xavier_normal_(self.z1.state_dict()['weight'])
        init.xavier_normal_(self.z2.state_dict()['weight'])
        init.xavier_normal_(self.next_mem.state_dict()['weight'])
        self.dropout = nn.Dropout(0.2)

    def make_interaction(self, facts, questions, prevM):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, 1, #hidden)
        prevM.size() -> (#batch, #sentence = 1, #hidden = #embedding)
        z.size() -> (#batch, #sentence, 4 x #embedding)
        G.size() -> (#batch, #sentence)
        '''

        B, F, D = facts.shape
        _, Q, _ = questions.shape

        facts = facts.unsqueeze(dim=1)  # B,1,F,D

        questions = questions.unsqueeze(dim=2)  # B,Q,1,D
        # print('make_interaction > questions  (expand) > ', tuple(questions.shape))
        prevM = prevM.unsqueeze(dim=2)  # B,Q,1,D
        # print('make_interaction > prevM > (expand) ', tuple(prevM.shape))

        z = torch.cat([
            facts * questions,
            facts * prevM,
            torch.abs(facts - questions),
            torch.abs(facts - prevM)
        ], dim=3)  # B, Q, F, 4D
        # print('make_interaction > z > ', tuple(z.shape))

        z = self.dropout(z)
        G = self.z2(self.dropout(torch.tanh(self.z1(z)))).squeeze(dim=3)  # B, Q, F
        att = torch.softmax(G, dim=-1)
        # print('make_interaction > att > ', tuple(att.shape))

        return att

    def forward(self, facts, questions, prevM):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, #sentence = 1, #hidden)
        prevM.size() -> (#batch, #sentence = 1, #hidden = #embedding)
        G.size() -> (#batch, #sentence)
        C.size() -> (#batch, #hidden)
        concat.size() -> (#batch, 3 x #embedding)
        '''
        # print('forward > facts > ', tuple(facts.shape))
        # print('forward > questions > ', tuple(questions.shape))
        # print('forward > prevM > ', tuple(prevM.shape))

        att = self.make_interaction(facts, questions, prevM)
        # print('forward > att > ', tuple(att.shape))

        C = self.AGRU(facts, att)
        # print('forward > C > ', tuple(C.shape))

        concat = torch.cat([prevM, C, questions], dim=-1)
        # print('forward > concat > ', tuple(concat.shape))
        concat = self.dropout(concat)

        next_mem = torch.tanh(self.next_mem(concat))
        # print('forward > next_mem > ', tuple(next_mem.shape))

        return next_mem


class DProto(FSLBaseModel):
    def __init__(self, encoder, args):
        super(DProto, self).__init__(encoder, args)
        self.dmn = EpisodicMemory(args.hidden_size, device=args.device)
        self.hop = 3

    def d_proto(self, support):
        """

        :param support: B, N, K, D
        :return:
        """
        B, N, K, D = support.shape
        support = torch.tanh(support.view(B * N, K, D))

        prototypes = torch.mean(support, dim=1, keepdim=True)
        ori_prototypes = prototypes
        for hop in range(self.hop):
            prototypes = self.dmn(support, ori_prototypes, prototypes)
        return prototypes

    def forward(self, batch, setting):
        _, N, K, Q = setting
        B = int(batch['length'].shape[0] / N / (K + Q))

        encoded = self.encoder(batch)
        embeddings = encoded['embedding']
        D = embeddings.shape[-1]
        embeddings = embeddings.view(B, N, K + Q, D)
        # print('| Proto > embedding', tuple(embeddings.shape))
        support = embeddings[:, :, :K, :].contiguous()  # B x N x K x D
        query = embeddings[:, :, K:, :].contiguous()  # B x N x Q x D

        prototypes = self.d_proto(support)
        query = torch.tanh(query)

        # print('Prototypes: ', tuple(prototypes.shape))

        unsqueeze_prototypes = prototypes.view(B, 1, N, D)  # B x 1 x N x D
        # print('| Proto > unsqueeze_prototypes', tuple(unsqueeze_prototypes.shape))

        query = query.view(B, N * Q, D)  # ->  B x NQ x D
        # print('| Proto > query', tuple(query.shape))

        query = query.unsqueeze(dim=2)  # ->  B x NQ x 1 x D
        # print('| Proto > query', tuple(query.shape))

        error = query - unsqueeze_prototypes  # B x NQ x N x D
        logits = -torch.sum(torch.pow(error, 2), dim=3)  # B x NQ x N

        # print(torch.mean(logits))

        return_item = {
            'logit': logits,
        }
        return return_item
