import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel


class BertEmbedding(nn.Module):

    def __init__(self, args):
        super(BertEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_pretrained)
        self.bert_layer = args.bert_layer

        for k, v in self.bert.named_parameters():
            v.requires_grad= args.tune_embedding


    def forward(self, inputs):
        # L = inputs['cls_text_sep_length'].max().cpu().numpy().tolist()
        # T = inputs['length'].max().cpu().numpy().tolist()

        text_bert_indices = inputs['cls_text_sep_indices']  # [:, :L]  # B  x L
        bert_segments_ids = inputs['cls_text_sep_segment_ids']  # [:, :L]  # B  x L
        transform = inputs['transform']  # [:, :T, :L].view(-1, T, L)  # B x T x L

        x, pooled_output = self.bert(text_bert_indices,
                                     bert_segments_ids,
                                     output_all_encoded_layers=True)
        # print('| BertGCN > x[0] ', tuple(x[0].shape))

        x = torch.cat(x[-self.bert_layer:], dim=-1)
        # print('| BertGCN > pooled_output ', tuple(pooled_output.shape))
        # print('| BertGCN > x ', tuple(x.shape))
        x = torch.bmm(transform, x)  # B x T x D
        return x
