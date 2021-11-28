import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_encoder.basenet import BaseNet, kl
from sentence_encoder.embedding import Embedding


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.nonlinear = nn.Tanh()

    def init_weight(self):
        self.apply(init_weights)

    def forward(self, text, adj):
        hidden = self.linear(text)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        return output


class GCN(BaseNet):
    def __init__(self, vectors, args):
        super(GCN, self).__init__(vectors, args)
        if args.tree == 'full':
            self.first_tree = 'dep'
            self.second_tree = 'prune_dep'
        else:
            self.first_tree = 'prune_dep'
            self.second_tree = 'dep'

        self.calculate_multual = True
        self.calculate_global_mutual = False

        self.embedder = Embedding(vectors,
                                  max_length=args.max_length,
                                  tune_embedding=args.tune_embedding,
                                  device=args.device)

        self.lstm = nn.LSTM(self.embedder.hidden_size, self.hidden_size, bidirectional=True, batch_first=True, num_layers=1)

        self.gc1 = GraphConvolution(2 * self.hidden_size, 2 * self.hidden_size)
        self.gc2 = GraphConvolution(2 * self.hidden_size, 2 * self.hidden_size)

        self.fc6 = nn.Sequential(
            nn.Dropout(), nn.Tanh(), nn.Linear(6 * self.hidden_size, self.hidden_size)#, nn.Tanh()
        )

        self.mutual = nn.Sequential(
            nn.Dropout(), nn.Sigmoid(), nn.Linear(2 * self.hidden_size, self.hidden_size), nn.Tanh()
        )
        self.max_pool = nn.MaxPool1d(kernel_size=args.max_length)

    def init_weight(self):
        self.gc1.init_weight()
        self.gc2.init_weight()

        self.gc1.apply(init_weights)
        self.gc2.apply(init_weights)
        self.lstm.apply(init_weights)

    def forward(self, inputs):
        # for k, v in inputs.items():
        #     print(k, v.shape)
        # exit(0)
        B = inputs['length'].shape[0]
        T = inputs['length'].max().cpu().numpy().tolist()

        anchor_index = inputs['anchor_index']  # BNKQ

        adj = inputs[self.first_tree][:, :T, :T]  # B x T x T

        # print('| BertGCN > x(bmm)', tuple(x.shape))
        # print('| BertGCN > inputs[sentence_length]', tuple(inputs['sentence_length'].shape))
        zero_one_two = torch.LongTensor([i for i in range(T)]).to(self.device).repeat(B, 1)  # B x T
        anchor_mask = zero_one_two == anchor_index.unsqueeze(1)
        anchor_mask = anchor_mask.unsqueeze(dim=2).expand(-1, -1, self.hidden_size * 2)
        x = self.embedder(inputs['indices'], inputs['dist'], inputs['mask'])

        lstm_x, _ = self.lstm(x[:, :T, :])

        # print('| BertGCN > x', tuple(x.shape))
        # print('| BertGCN > anchor_mask', tuple(anchor_mask.shape))
        # print('| BertGCN > mask', tuple(mask.shape))
        lstm_anchor = torch.masked_select(lstm_x, anchor_mask).view(B, -1)

        x = self.gc1(lstm_x, adj)
        gcn_anchor_1 = torch.masked_select(x, anchor_mask).view(B, -1)
        x = self.dropout(x)
        x = self.gc2(x, adj)

        gcn_anchor_2 = torch.masked_select(x, anchor_mask).view(B, -1)

        rep = torch.cat([lstm_anchor, gcn_anchor_1, gcn_anchor_2], dim=-1)
        rep = self.fc6(rep)

        # rep = torch.cat([lstm_anchor, gcn_anchor_2], dim=-1)
        # rep = self.fc4(rep)

        # Structure mutual between full tree and prune tree
        adj2 = inputs[self.second_tree][:, :T, :T]  # B x T x T
        x2 = self.gc1(lstm_x, adj2)
        x2 = self.dropout(x2)
        x2 = self.gc2(x2, adj2)

        # print('| BertGCN > x', tuple(x.shape))
        # print('| BertGCN > x2', tuple(x2.shape))

        x = torch.transpose(x, dim1=1, dim0=2)
        x2 = torch.transpose(x2, dim1=1, dim0=2)

        # print('| BertGCN > x', tuple(x.shape))
        # print('| BertGCN > x2', tuple(x2.shape))
        pool_x = F.max_pool1d(x, T).squeeze(dim=2)
        pool_x2 = F.max_pool1d(x2, T).squeeze(dim=2)

        # print('| BertGCN > pool_x', tuple(pool_x.shape))
        # print('| BertGCN > pool_x2', tuple(pool_x2.shape))

        # gcn_anchor_2 = torch.masked_select(x2, anchor_mask).view(B, -1)
        m1 = self.mutual(pool_x)
        m2 = self.mutual(pool_x2)

        mutual_loss = kl(m1, m2)
        # mutual_loss = self.kl(p1, p2)

        return_item = {
            'embedding': rep,
            'mutual_loss': mutual_loss,
            'pool1': pool_x,
            'pool2': pool_x2
        }

        return return_item
