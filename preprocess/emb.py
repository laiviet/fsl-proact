import transformers
import argparse
import sys
import tqdm
from preprocess.utils import *
import numpy as np

os.environ['TRANSFORMERS_CACHE'] = 'cache'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from torch.utils.data import Dataset, DataLoader

from pytorch_pretrained_bert import BertTokenizer, BertModel

BERT_VERSION = 'bert-base-cased'

ML = 128

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if not torch.cuda.is_available():
    print('| WARNING: No cuda device detected')

tokenizer = BertTokenizer.from_pretrained(BERT_VERSION, do_lower_case=False)

SEP = 102
CLS = 101
PAD = 0


class RAMSDataset(Dataset):

    def __init__(self, prefix):
        super(RAMSDataset, self).__init__()

        self.max_len = 40
        self.bert_max_len = 200
        # Load data
        self.raw = load_json('{}.prune.json'.format(prefix))
        self.indices = load_json('{}.{}.json'.format(prefix, BERT_VERSION))
        self.cache = {}
        print('Check id matching', end=' ')
        assert len(self.raw) == len(self.indices), 'Raw: {}, Indices: {}'.format(len(self.raw), len(self.indices))
        for r, b in zip(self.raw, self.indices):
            assert r['id'] == b['id'], 'Raw and  BERT mismatch'
        print('#Instance: ', len(self.raw), len(self.indices))

        # for i in range(len(self.raw)):
        #     self.cache[i] = self.preprocess(i)

    def preprocess(self, i):
        raw = self.raw[i]

        indices = self.indices[i]
        preprocesed = {
            'id': raw['id']
        }
        ml = self.max_len
        bl = self.bert_max_len

        crop_bert_indices = indices['bert_indices']
        cls_text_sep = [CLS]
        transform = np.zeros(shape=(ml, bl))
        offset = 1
        for i, ids in enumerate(crop_bert_indices):
            l = len(ids)
            cls_text_sep += ids
            if l == 0:
                continue
            transform[i, offset:offset + l] = 1.0 / l
            offset += l
        cls_text_sep.append(SEP)
        cls_text_sep_length = len(cls_text_sep)
        assert cls_text_sep_length <= bl, 'Exceed bert length: {} vs {}'.format(cls_text_sep_length, bl)
        cls_text_sep += [PAD for _ in range(bl - cls_text_sep_length)]

        preprocesed['cls_text_sep_indices'] = cls_text_sep
        preprocesed['cls_text_sep_length'] = [cls_text_sep_length]
        preprocesed['cls_text_sep_segment_ids'] = [1 for _ in range(cls_text_sep_length)] + [0 for _ in range(
            bl - cls_text_sep_length)]
        preprocesed['transform'] = transform

        ai = raw['trigger'][0]
        preprocesed['anchor_index'] = ai

        return preprocesed

    def get_one(self, i):
        if i in self.cache:
            return self.cache[i]
        else:
            item = self.preprocess(i)
            self.cache[i] = item
            return item

    @staticmethod
    def sup_pack(items):
        batches = {}
        for fea in items[0].keys():
            data = [x[fea] for x in items]
            if fea in FeatureTensor:
                batches[fea] = FeatureTensor[fea](data)
            else:
                batches[fea] = data
        return batches

    def __getitem__(self, item):
        return self.get_one(item)

    def __len__(self):
        return len(self.raw)


FeatureTensor = {
    'indices': torch.LongTensor,
    'length': torch.LongTensor,
    'dist': torch.LongTensor,
    'anchor_index': torch.LongTensor,
    'mask': torch.FloatTensor,
    # BERT
    'cls_text_sep_indices': torch.LongTensor,
    'cls_text_sep_length': torch.LongTensor,
    'cls_text_sep_segment_ids': torch.LongTensor,
    'transform': torch.FloatTensor,

    # Semcor
    'label_mask': torch.FloatTensor,
    # Precompute BERT
    'emb': torch.FloatTensor
}


class BertEmbedding(torch.nn.Module):

    def __init__(self):
        super(BertEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_VERSION)
        self.bert_layer = 12

    def select_anchor(self, emb, anchor_index):
        """

        :param emb: B x L x D
        :param anchor_index: B
        :return:
        """
        B, L, D = emb.shape
        u = torch.tensor([x for x in range(L)]).unsqueeze(0).to(device)
        v = anchor_index.view(B, 1)
        mask = (u == v).unsqueeze(dim=2).to(device)
        x = torch.masked_select(emb, mask).view(-1, D)
        return x

    def forward(self, inputs):
        # L = inputs['cls_text_sep_length'].max().cpu().numpy().tolist()
        # T = inputs['length'].max().cpu().numpy().tolist()

        text_bert_indices = inputs['cls_text_sep_indices'].to(device)  # [:, :L]  # B  x L
        bert_segments_ids = inputs['cls_text_sep_segment_ids'].to(device)  # [:, :L]  # B  x L
        transform = inputs['transform'].to(device)  # [:, :T, :L].view(-1, T, L)  # B x T x L

        x, pooled_output = self.bert(text_bert_indices,
                                     bert_segments_ids,
                                     output_all_encoded_layers=True)
        # print('| BertGCN > x[0] ', tuple(x[0].shape))

        x = torch.cat(x[-self.bert_layer:], dim=-1)
        # print('| BertGCN > pooled_output ', tuple(pooled_output.shape))
        # print('| BertGCN > x ', tuple(x.shape))
        x = torch.bmm(transform, x)  # B x T x D

        x = self.select_anchor(x, inputs['anchor_index'].to(device))

        x = x.view(x.shape[0], self.bert_layer, 768)

        return x


def do_emb(prefix):
    dataset = RAMSDataset(prefix)
    dl = DataLoader(dataset, batch_size=256, collate_fn=RAMSDataset.sup_pack, num_workers=4)
    bert_model = BertEmbedding().to(device)

    save_embedding = []
    with torch.no_grad():
        for batch in dl:
            trigger_emb = bert_model(batch)

            for id, emb in zip(batch['id'], trigger_emb):
                # print(emb.shape)
                save_embedding.append({
                    'id': id,
                    'emb': emb.cpu().numpy().tolist()
                })
    print('Total: ', len(dataset))
    save_pickle(save_embedding, '{}.bert.pkl'.format(prefix))


if __name__ == '__main__':

    print(sys.argv)

    # if sys.argv[-1] == "1":
    #     do_emb('datasets/rams/fsl/train')
    # if sys.argv[-1] == "2":
    #     do_emb('datasets/rams/fsl/test')
    #     do_emb('datasets/rams/fsl/test.negative')
    # if sys.argv[-1] == "3":
    #     do_emb('datasets/rams/fsl/dev')
    #     do_emb('datasets/rams/fsl/dev.negative')
    # if sys.argv[-1] == "0":
    #     do_emb('datasets/rams/fsl/train.negative')
    if sys.argv[-1] == "1":
        do_emb('datasets/ace/fsl/test')
        do_emb('datasets/ace/fsl/dev')
    if sys.argv[-1] == "2":
        do_emb('datasets/ace/fsl/train')
    if sys.argv[-1] == "3":
        do_emb('datasets/ace/fsl/test.negative')
        do_emb('datasets/ace/fsl/dev.negative')
    if sys.argv[-1] == "0":
        do_emb('datasets/ace/fsl/train.negative')
    #
    # if sys.argv[-1] == "3":
    #     do_emb('datasets/fed/fsl/test')
    #     do_emb('datasets/fed/fsl/dev')
    #     do_emb('datasets/fed/fsl/train')
    #     do_emb('datasets/fed/fsl/test.negative')
    #     do_emb('datasets/fed/fsl/dev.negative')
    #     do_emb('datasets/fed/fsl/train.negative')
