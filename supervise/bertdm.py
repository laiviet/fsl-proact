import torch
import torch.nn as nn
from sentence_encoder.bert_base import BertEmbedding
from sentence_encoder.basenet import init_weights


class BertDMClassifier(nn.Module):
    def __init__(self, vectors, args):
        super(BertDMClassifier, self).__init__()
        self.device = args.device
        self.bert_layer = args.bert_layer
        self.bert = BertEmbedding(args)
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.bert_layer * 768 * 3, args.n_class)

        self.hidden_size = args.hidden_size

        self.m = torch.LongTensor([i for i in range(200)]).to(self.device)

    def init_weight(self):
        self.fc.apply(init_weights)

    def get_mask(self, L, anchor_index, length):
        m = self.m[:L].repeat(anchor_index.shape[0], 1)
        anchor_index = anchor_index.unsqueeze(dim=1)

        maskL = m < (anchor_index + 1)
        maskR = m > anchor_index

        maskL = maskL.float().unsqueeze(dim=0)
        maskR = maskR.float().unsqueeze(dim=0)

        maskS = m < length.unsqueeze(dim=1)
        maskS = maskS.float().unsqueeze(dim=0)
        maskR = maskR * maskS
        return maskL, maskR

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

    def forward(self, inputs):
        """
        bert_length: L
        original_length: T
        """

        L = inputs['cls_text_sep_length'].max().cpu().numpy().tolist()
        T = inputs['length'].max().cpu().numpy().tolist()
        sentence_length = inputs['length']
        anchor_index = inputs['anchor_index']

        embedding = self.bert(inputs)[:,:T,:]   # B,T,D
        transpose_x = embedding.transpose(1, 2).transpose(0, 1)

        # print('| BertDM > anchor_index', tuple(anchor_index.shape))
        # print('| BertDM > sentence_length', tuple(sentence_length.shape))
        # print('| BertDM > embedding', tuple(embedding.shape))


        maskL, maskR = self.get_mask(T, anchor_index, sentence_length)

        # print('| BertDM > transpose_x', tuple(transpose_x.shape))
        # print('| BertDM > maskL', tuple(maskL.shape))
        # print('| BertDM > maskR', tuple(maskR.shape))


        L = (transpose_x * maskL).transpose(0, 1)
        R = (transpose_x * maskR).transpose(0, 1)
        # print('| BertDM > L', tuple(L.shape))
        # print('| BertDM > R', tuple(R.shape))

        L = L + 1
        R = R + 1
        # print('| BertDM > L', tuple(L.shape))

        pooledL, _ = L.max(dim=2)
        pooledR, _ = R.max(dim=2)
        x = torch.cat((pooledL, pooledR), 1)
        x = x - 1

        anchor_rep = self.select_anchor(embedding, anchor_index)
        # print('| BertDM > anchor_rep', tuple(anchor_rep.shape))

        anchor_rep = self.dropout(anchor_rep)
        x = torch.cat((anchor_rep, x), 1)
        logits = self.fc(self.dropout(x))

        return {
            'logit': logits
        }
