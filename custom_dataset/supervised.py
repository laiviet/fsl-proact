import json
from torch.utils.data import Dataset, DataLoader
from preprocess.tokenizer import *
from preprocess.utils import *
import random
import torch
import numpy as np
import os


def merge(r, g, b):
    item = {}
    for k, v in r.items():
        item[k] = v
    for k, v in g.items():
        item[k] = v
    for k, v in b.items():
        item[k] = v
    return item


class RAMSDataset(Dataset):

    def __init__(self,
                 features=(),
                 prefix='datasets/ace/fsl/train',
                 bert_pretrain=BERT_BASE_CASED,
                 device='cpu',
                 load_negative=True):
        super(RAMSDataset, self).__init__()

        self.max_len = 40
        self.device = device
        self.features = features
        prefix_parts = prefix.split('/')
        label_map_path = 'datasets/{}/{}/label_map.json'.format(
            prefix_parts[1], prefix_parts[2]
        )
        assert len(prefix_parts) == 4

        self.bert_max_len = 200

        # Load label, ner, pos, dep, argument map
        m = load_json(label_map_path)
        self.label_map = m['label']
        self.n_class = len(self.label_map)
        # self.ner_map = m['stanford_ner']
        # self.pos_map = m['stanford_pos']
        # self.dep_map = m['standord_dependency']
        self.argument_map = m['argument']

        # Load data
        self.raw = load_json('{}.prune.json'.format(prefix))
        self.indices = load_json('{}.{}.json'.format(prefix, bert_pretrain))

        if load_negative:
            self.raw += load_json('{}.negative.prune.json'.format(prefix))
            self.indices += load_json('{}.negative.{}.json'.format(prefix, bert_pretrain))

        print('Load')

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
            'i': i,
            'target': self.label_map[str(raw['label'])]
        }
        ml = self.max_len
        bl = self.bert_max_len

        if 'cls_text_sep_indices' in self.features:
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

        if 'anchor_index' in self.features:
            ai = raw['trigger'][0]
            preprocesed['anchor_index'] = ai
            preprocesed['dist'] = [i - ai + 40 for i in range(ml)]

            assert all([i >= 0 for i in preprocesed['dist']]), 'Wrong dist'

        glove_indices = indices['glove_indices']
        if 'indices' in self.features:
            l = len(glove_indices)
            assert l <= ml, 'Wrong length: {}'.format(l)
            pad_glove_indices = glove_indices + [0 for _ in range(ml - l)]
            assert len(pad_glove_indices) == ml, 'Wrong glove indices, '
            preprocesed['indices'] = pad_glove_indices
            preprocesed['length'] = l
            preprocesed['mask'] = [1.0 for x in range(l)] + [0.0 for _ in range(ml - l)]

            indices_idx = set()
            prune_footprint = [0.0 for x in range(ml)]

            for h, t, label in raw['edge_prune']:
                indices_idx.update([h, t])
                prune_footprint[h] = 1.0
                prune_footprint[t] = 1.0
            indices_idx = sorted(indices_idx)
            l = len(indices_idx)
            prune_indices = [glove_indices[x] for x in indices_idx]
            prune_indices += [0 for _ in range(ml - l)]
            preprocesed['prune_indices'] = prune_indices
            preprocesed['prune_length'] = l
            preprocesed['prune_mask'] = [1.0 for _ in range(l)] + [0.0 for _ in range(ml - l)]

            preprocesed['prune_footprint'] = prune_footprint
        else:
            # print('Ignore indices feature')
            pass

        if 'entity_indices' in self.features:
            entity_attention = np.ones((ml))
            entity_indices = [self.argument_map['Other'] for x in range(ml)]
            for start, end, type in raw['argument']:
                for i in range(start, end + 1):
                    if i < ml:
                        # A very big number because we use 1 as default (to avoid divide by 0 when normalizing)
                        entity_attention[i] = 100000.0
                        entity_indices[i] = self.argument_map[type]
            entity_attention = entity_attention / entity_attention.sum()
            preprocesed['entity_indices'] = entity_indices
            preprocesed['entity_attention'] = entity_attention

        if 'dep' in self.features:
            dep_matrix = np.eye(ml)
            for h, t, label in raw['edge']:
                dep_matrix[h][t] = 1.0
                dep_matrix[t][h] = 1.0
            preprocesed['dep'] = dep_matrix
            # Prune tree
            dep_matrix = np.eye(ml)
            for h, t, label in raw['edge_prune']:
                dep_matrix[h][t] = 1.0
                dep_matrix[t][h] = 1.0
            preprocesed['prune_dep'] = dep_matrix

        return preprocesed

    def get_one(self, i):
        if i in self.cache:
            return self.cache[i]
        else:
            item = self.preprocess(i)
            self.cache[i] = item
            return item

    def __getitem__(self, item):
        return self.get_one(item)

    def __len__(self):
        return len(self.raw)


class FSLDataset(RAMSDataset):
    def __init__(self, N, K, Q, length, **kwargs):
        super(FSLDataset, self).__init__(**kwargs)
        self.N = N
        self.K = K
        self.Q = Q
        self.length = length

        labels = [x['label'] for x in self.raw]
        label_set = sorted(set(labels))
        self.fsl_label_map = {l: i for i, l in enumerate(label_set)}
        # Just a list of [1,2,...,len(available_labels)-1]
        # for x in sorted(label_set):
        #     print(x)
        self.positive_targets = [i for i in range(1, len(label_set))]

        self.label_indices_map = [[] for _ in range(len(label_set))]

        for i, label in enumerate(labels):
            fsl_target = self.fsl_label_map[label]
            self.label_indices_map[fsl_target].append(i)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        # random.seed(item)
        selected_fsl_target = [0] + random.sample(self.positive_targets, k=self.N)
        all_class_indices = []
        targets = []
        sample_per_class = self.K + self.Q

        for i, target in enumerate(selected_fsl_target):
            possible_indices = self.label_indices_map[target]
            sampled_indices = random.sample(possible_indices, k=sample_per_class)
            all_class_indices += sampled_indices
            targets.append([i for _ in range(self.Q)])
        all_class_data = [self.get_one(x) for x in all_class_indices]
        batch = dict()
        for fea in self.features:
            data = [x[fea] for x in all_class_data]
            try:
                batch[fea] = FeatureTensor[fea](data)
            except:
                print(fea)
                for x in data:
                    print(len(x))
                exit(0)

        batch['target'] = torch.LongTensor(targets)
        return batch


def sup_pack(items):
    batches = {}
    for fea in items[0].keys():
        data = [x[fea] for x in items]
        batches[fea] = FeatureTensor[fea](data)
    return batches


def fsl_pack(items):
    batches = {}
    for fea in items[0].keys():
        data = [x[fea] for x in items]
        # batches[fea] = FeatureTensor[fea](data)
        batches[fea] = torch.cat(data, dim=0)
        # print(fea, batches[fea].shape)
    # print('------')
    return batches


def keep(x):
    return x


FeatureTensor = {
    'i': torch.LongTensor,  # Index of the sample
    # ANN
    'entity_indices': torch.LongTensor,
    'entity_attention': torch.FloatTensor,
    # CNN, LSTM, GRU
    'indices': torch.LongTensor,
    'prune_indices': torch.LongTensor,
    'length': torch.LongTensor,
    'prune_length': torch.LongTensor,
    'dist': torch.LongTensor,
    'anchor_index': torch.LongTensor,
    'prune_anchor_index': torch.LongTensor,
    'mask': torch.FloatTensor,
    'prune_mask': torch.FloatTensor,
    'prune_footprint': torch.FloatTensor,
    # BERT
    'cls_text_sep_indices': torch.LongTensor,
    'cls_text_sep_length': torch.LongTensor,
    'cls_text_sep_segment_ids': torch.LongTensor,
    'transform': torch.FloatTensor,
    # GCN
    'dep': torch.FloatTensor,
    'prune_dep': torch.FloatTensor,
    'target': torch.LongTensor,
    # Semcor
    'label_mask': torch.FloatTensor,
    # Precompute BERT
    'emb': torch.FloatTensor
}


class SemcorDataset(Dataset):

    def __init__(self,
                 features=(),
                 prefix='datasets/ace/fsl/train',
                 bert_pretrain=BERT_BASE_CASED,
                 device='cpu'):
        super(SemcorDataset, self).__init__()

        self.max_len = 40
        self.device = device
        self.features = features

        prefix_parts = prefix.split('/')
        label_map_path = 'datasets/{}/{}/label_map.json'.format(
            prefix_parts[1], prefix_parts[2]
        )
        assert len(prefix_parts) == 4

        self.bert_max_len = 128

        # Load label, ner, pos, dep, argument map
        m = load_json(label_map_path)
        self.label_map = m['label']
        self.n_class = len(self.label_map)

        # Load data
        self.original = load_json('{}.json'.format(prefix))
        self.original += load_json('{}.negative.json'.format(prefix))
        self.raw = load_json('{}.prune.json'.format(prefix))
        self.raw += load_json('{}.prune.negative.json'.format(prefix))
        self.indices = load_json('{}.{}.json'.format(prefix, bert_pretrain))
        self.indices += load_json('{}.{}.negative.json'.format(prefix, bert_pretrain))

        self.cache = {}
        # print('Check id matching', end=' ')
        assert len(self.raw) == len(self.indices), 'Raw: {}, Indices: {}'.format(len(self.raw), len(self.indices))
        for r, b in zip(self.raw, self.indices):
            assert r['id'] == b['id'], 'Raw and  BERT mismatch'
        # print('PASS')

        # print('#Instance: ', len(self.raw), len(self.indices))

        # for i in range(len(self.raw)):
        #     self.cache[i] = self.preprocess(self.original[i], self.raw[i], self.indices[i])

    def __len__(self):
        return len(self.raw)

    def preprocess(self, original, raw, indices):
        preprocesed = {
            'target': self.label_map[raw['label']]
        }
        ml = self.max_len
        bl = self.bert_max_len
        ai = raw['trigger'][0]

        if 'cls_text_sep_indices' in self.features:
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
        if 'bert_transform':
            transform = np.zeros(shape=(ml, bl))
            offset = 1
            for i, ids in enumerate(indices['bert_indices']):
                l = len(ids)
                if l == 0:
                    continue
                transform[i, offset:offset + l] = 1.0 / l
                offset += l
            preprocesed['transform'] = transform
        if 'bert_anchor_mask':
            mask = np.zeros(shape=(bl))
            offset = 1
            for ids in indices['bert_indices'][:ai]:
                offset += len(ids)
            mask[offset:offset + len(indices['bert_indices'][ai])] = 1.0

        if 'anchor_index' in self.features:
            preprocesed['anchor_index'] = ai
            preprocesed['dist'] = [i - ai + 40 for i in range(ml)]

            assert all([i >= 0 for i in preprocesed['dist']]), 'Wrong dist'

        glove_indices = indices['glove_indices']
        if 'indices' in self.features:
            l = len(glove_indices)
            assert l <= ml, 'Wrong length: {}'.format(l)
            pad_glove_indices = glove_indices + [0 for _ in range(ml - l)]
            assert len(pad_glove_indices) == ml, 'Wrong glove indices, '
            preprocesed['indices'] = pad_glove_indices
            preprocesed['length'] = len(glove_indices)
            preprocesed['mask'] = [1.0 for x in range(l)] + [0.0 for _ in range(ml - l)]

            indices_idx = set()
            prune_footprint = [0.0 for x in range(ml)]

            for h, t, label in raw['edge_prune']:
                indices_idx.update([h, t])
                prune_footprint[h] = 1.0
                prune_footprint[t] = 1.0
            indices_idx = sorted(indices_idx)
            l = len(indices_idx)
            prune_indices = [glove_indices[x] for x in indices_idx]
            prune_indices += [0 for _ in range(ml - l)]
            preprocesed['prune_indices'] = prune_indices
            preprocesed['prune_length'] = l
            preprocesed['prune_mask'] = [1.0 for _ in range(l)] + [0.0 for _ in range(ml - l)]

            preprocesed['prune_footprint'] = prune_footprint

        if 'label_mask' in self.features:
            label_mask = [0.0 for _ in range(self.n_class)]
            for c in original['candidate']:
                label_mask[self.label_map[c]] = 1.0
            preprocesed['label_mask'] = label_mask

        if 'dep' in self.features:
            dep_matrix = np.eye(ml)
            for h, t, label in raw['edge']:
                dep_matrix[h][t] = 1.0
                dep_matrix[t][h] = 1.0
            preprocesed['dep'] = dep_matrix

            # Prune
            dep_matrix = np.eye(ml)
            for h, t, label in raw['edge_prune']:
                dep_matrix[h][t] = 1.0
                dep_matrix[t][h] = 1.0
            preprocesed['prune_dep'] = dep_matrix
        return preprocesed

    def __getitem__(self, i):
        if i in self.cache:
            return self.cache[i]
        else:
            item = self.preprocess(self.original[i], self.raw[i], self.indices[i])
            self.cache[i] = item
            return item

    @staticmethod
    def pack(items):
        batches = {}
        for fea in items[0].keys():
            data = [x[fea] for x in items]
            batches[fea] = FeatureTensor[fea](data)
        return batches


def test_dataset():
    dataset = RAMSDataset(prefix='datasets/ace/fsl/train')
    dataloader = DataLoader(dataset, batch_size=5)

    for i, batch in enumerate(dataloader):
        print(i)
        for k, v in batch.items():
            print(k, v.shape)


if __name__ == '__main__':
    test_dataset()
