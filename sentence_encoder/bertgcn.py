import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel

from sentence_encoder.basenet import kl


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


class BertGCN(nn.Module):
    def __init__(self, vectors, args):
        super(BertGCN, self).__init__()
        # self.squeeze_embedding = SqueezeEmbedding()
        self.device = args.device
        self.bert = BertModel.from_pretrained(args.bert_pretrained)
        self.dropout = nn.Dropout(args.dropout)
        self.hidden_size = hidden_size = args.hidden_size
        self.bert_layer = args.bert_layer

        self.lstm = nn.LSTM(768 * self.bert_layer, 2 * hidden_size, bidirectional=True, batch_first=True, num_layers=1)

        if args.tree == 'full':
            self.first_tree = 'dep'
            self.second_tree = 'prune_dep'
        else:
            self.first_tree = 'prune_dep'
            self.second_tree = 'dep'

        self.gc1 = GraphConvolution(2 * 2 * self.hidden_size, 2 * 2 * self.hidden_size)
        self.gc2 = GraphConvolution(2 * 2 * self.hidden_size, 2 * 2 * self.hidden_size)

        self.fc = nn.Sequential(
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(4 * 2 * 2 * hidden_size + 768 * self.bert_layer, self.hidden_size)
        )

        self.mutual = nn.Sequential(
            nn.Dropout(), nn.Sigmoid(), nn.Linear(2 * 2 * self.hidden_size, self.hidden_size), nn.Tanh()
        )
        self.max_pool = nn.MaxPool1d(kernel_size=args.max_length)

        for params in self.bert.parameters():
            params.requires_grad = args.bert_update

    def init_weight(self):
        self.gc1.init_weight()
        self.gc2.init_weight()

        self.gc1.apply(init_weights)
        self.gc2.apply(init_weights)
        self.lstm.apply(init_weights)



    def forward(self, inputs):
        """
        bert_length: L
        original_length: T
        """

        B = inputs['length'].shape[0]

        L = inputs['cls_text_sep_length'].max().cpu().numpy().tolist()
        T = inputs['length'].max().cpu().numpy().tolist()

        text_bert_indices = inputs['cls_text_sep_indices'][:, :L]  # B  x L
        bert_segments_ids = inputs['cls_text_sep_segment_ids'][:, :L]  # B  x L
        transform = inputs['transform'][:, :T, :L].view(-1, T, L)  # B x T x L
        adj1 = inputs[self.first_tree][:, :T, :T]  # B x T x T
        anchor_index = inputs['anchor_index']  # B

        x, pooled_output = self.bert(text_bert_indices,
                                     bert_segments_ids,
                                     output_all_encoded_layers=True)
        # print('| BertGCN > x[0] ', tuple(x[0].shape))

        x = torch.cat(x[-self.bert_layer:], dim=-1)
        # print('| BertGCN > pooled_output ', tuple(pooled_output.shape))
        # print('| BertGCN > x ', tuple(x.shape))
        x = torch.bmm(transform, x)  # B x T x D

        # print('| BertGCN > x(bmm)', tuple(x.shape))
        # print('| BertGCN > inputs[sentence_length]', tuple(inputs['sentence_length'].shape))
        mask = torch.LongTensor([i for i in range(T)]).cuda().repeat(B, 1)  # B x T
        mask = mask == anchor_index.unsqueeze(1)
        bert_mask = mask.unsqueeze(dim=2).expand(-1, -1, 768 * self.bert_layer)
        anchor_rep = torch.masked_select(x, bert_mask).view(B, -1)
        anchor_rep = self.dropout(anchor_rep)

        self.lstm.flatten_parameters()
        lstm_x, _ = self.lstm(x)

        # print('| BertGCN > anchor_index', tuple(anchor_index.shape))

        lstm_mask = mask.unsqueeze(dim=2).expand(-1, -1, self.hidden_size * 2 * 2)
        # print('| BertGCN > mask', tuple(mask.shape))

        # print('| BertGCN > aspect', tuple(aspect.shape))

        #
        # gate1 = self.dropout(gate1)
        # gate2 = self.dropout(gate2)
        gcn1_x = self.gc1(lstm_x, adj1)
        gcn2_x = self.gc2(gcn1_x, adj1)

        aspect_lstm = torch.masked_select(lstm_x, lstm_mask).view(B, -1)
        aspect_gcn1 = torch.masked_select(gcn1_x, lstm_mask).view(B, -1)
        aspect_gcn2 = torch.masked_select(gcn2_x, lstm_mask).view(B, -1)

        out = torch.max(gcn2_x, dim=1)[0]
        out = self.dropout(out)
        rep = self.fc(torch.cat([anchor_rep, aspect_lstm, aspect_gcn1, aspect_gcn2, out], dim=1))

        # Structure mutual between full tree and prune tree
        adj2 = inputs[self.second_tree][:, :T, :T]  # B x T x T
        x2 = self.gc1(lstm_x, adj2)
        x2 = self.dropout(x2)
        x2 = self.gc2(x2, adj2)

        # print('| BertGCN > x', tuple(x.shape))
        # print('| BertGCN > x2', tuple(x2.shape))

        x = torch.transpose(gcn2_x, dim1=1, dim0=2)
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
