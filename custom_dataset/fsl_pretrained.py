from preprocess.utils import *
import torch
import random
from torch.utils.data import Dataset
import numpy as np


class EmbeddingFSLDataset(Dataset):

    def __init__(self, N, K, Q, length, prefix, bert_layer=4):
        super(EmbeddingFSLDataset, self).__init__()
        self.N = N
        self.K = K
        self.Q = Q
        self.length = length
        self.bert_layer = bert_layer

        self.raw = load_json('{}.prune.json'.format(prefix))
        self.raw += load_json('{}.negative.prune.json'.format(prefix))

        embedding = load_pickle('{}.bert.pkl'.format(prefix))
        embedding += load_pickle('{}.negative.bert.pkl'.format(prefix))

        print('#Instance: ', len(self.raw), len(embedding))
        for r, b in zip(self.raw, embedding):
            assert r['id'] == b['id'], 'Raw and  BERT mismatch'

        embedding = [x['emb'] for x in embedding]
        self.embedding = torch.FloatTensor(embedding)
        n = self.embedding.shape[0]
        self.embedding = self.embedding[:, -self.bert_layer:, :].view(n, -1).contiguous()

        print('Embedding: ', self.embedding.shape)

        labels = [x['label'] for x in self.raw]
        label_set = sorted(set(labels))

        self.fsl_label_map = {l: i for i, l in enumerate(label_set)}
        # Just a list of [1,2,...,len(available_labels)-1]
        # for x in sorted(label_set):
        #     print(x)

        self.positive_targets = [i for i in range(1, len(label_set))]

        self.label_indices_map = [[] for _ in range(len(label_set))]
        for i, label in enumerate(labels):  # list of label of each item
            fsl_target = self.fsl_label_map[label]
            self.label_indices_map[fsl_target].append(i)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        # random.seed(item)
        selected_fsl_target = [0] + random.sample(self.positive_targets, k=self.N)
        sample_per_class = self.K + self.Q

        support_set, support_targets, query_set, query_targets = [], [], [], []

        for i, target in enumerate(selected_fsl_target):
            possible_indices = self.label_indices_map[target]
            if len(possible_indices) < sample_per_class:
                possible_indices = possible_indices + possible_indices + possible_indices
            sampled_indices = random.sample(possible_indices, k=sample_per_class)
            support_set.append(sampled_indices[:self.K])
            query_set.append(sampled_indices[self.K:])
            support_targets.append([i for _ in range(self.K)])
            query_targets.append([i for _ in range(self.Q)])
        batch = {
            'support_set': torch.LongTensor(support_set),
            'support_targets': torch.LongTensor(support_targets),
            'query_set': torch.LongTensor(query_set),
            'query_targets': torch.LongTensor(query_targets)
        }
        return batch

    @staticmethod
    def nopack(items):
        return items

    @staticmethod
    def fsl_pack(items):
        batches = {}
        for fea in items[0].keys():
            data = [x[fea] for x in items]
            # batches[fea] = FeatureTensor[fea](data)
            batches[fea] = torch.stack(data, dim=0)
            # print(fea, batches[fea].shape)
        # print('------')
        return batches


FeatureTensor = {
    'i': torch.LongTensor,
    'target': torch.LongTensor,
}
