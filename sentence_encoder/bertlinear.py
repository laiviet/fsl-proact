import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class BertLinear(nn.Module):

    def __init__(self, vectors=None, args=None):
        super(BertLinear, self).__init__()
        self.device = args.device
        self.bert = BertModel.from_pretrained(args.bert_pretrained)
        self.dropout = nn.Dropout(args.dropout)
        self.hidden_size = args.hidden_size
        self.n_layer = 12

        for params in self.bert.parameters():
            params.requires_grad = False

        self.fc1 = nn.Sequential(
            nn.Tanh(),
            nn.Linear(768 * self.n_layer, args.hidden_size),
        )

    def init_weight(self):
        self.fc1.apply(init_weights)

    def forward(self, inputs):
        L = inputs['cls_text_sep_length'].max()
        T = inputs['length'].max()

        text_bert_indices = inputs['cls_text_sep_indices'][:, :L]  # B x NKQ x L
        bert_segments_ids = inputs['cls_text_sep_segment_ids'][:, :L]  # B x NKQ x L
        transform = inputs['transform'][:, :T, :L]  # B x NKQ x T x L
        anchor_index = inputs['anchor_index']  # BNKQ
        B = anchor_index.shape[0]

        x, pooled_output = self.bert(text_bert_indices.contiguous().view(-1, L),
                                     bert_segments_ids.contiguous().view(-1, L),
                                     output_all_encoded_layers=True)
        x = torch.cat(x, dim=-1)
        x = torch.bmm(transform, x)  # B x T x D

        mask = torch.LongTensor([i for i in range(T)]).to(self.device).repeat(B, 1)  # B x T
        mask = mask == anchor_index.unsqueeze(1)
        bert_mask = mask.unsqueeze(dim=2).expand(-1, -1, 768 * self.n_layer)
        anchor_rep = torch.masked_select(x, bert_mask).view(B, -1)
        anchor_rep = self.dropout(anchor_rep)

        rep = self.fc1(anchor_rep)

        return {
            'embedding': rep
        }
