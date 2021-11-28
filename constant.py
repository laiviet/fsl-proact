import transformers
from preprocess.tokenizer import *
from sentence_encoder import *
from fewshot import *
from supervise import *
import tqdm

dataset_constant = {
    'ace': {
        'max_length': 40,
        'n_class': 34,
        'other': 1,  # Means has other class, doing event detection
        'train_way': 18,
    },
    'fed': {
        'max_length': 40,
        'n_class': 450,
        'other': 1,  # Means has other class, doing event detection
        'train_way': 20,
    },
    'rams': {
        'max_length': 40,
        'n_class': 140,
        'other': 1,  # Means has other class, doing event detection
        'train_way': 20,
    },
    'rams2': {
        'max_length': 40,
        'n_class': 39,
        'other': 1,  # Means has other class, doing event detection
        'train_way': 20,
    },
    'semcor': {
        'max_length': 40,
        'n_class': 7435,
        'other': 1,  # Means has other class, doing event detection
        'train_way': 20,
    },
    'cyber': {
        'max_length': 40,
        'n_class': 32,
        'other': 1,  # Means has other class, doing event detection
        'train_way': 20,
    },
    'lowkbp': {
        'max_length': 40,
        'n_class': 92,
        'other': 0,  # Means no other, doing event classification
        'train_way': 18,
    }
}

dataset_constant['lowkbp0'] = dataset_constant['lowkbp']
dataset_constant['lowkbp1'] = dataset_constant['lowkbp']
dataset_constant['lowkbp2'] = dataset_constant['lowkbp']
dataset_constant['lowkbp3'] = dataset_constant['lowkbp']
dataset_constant['lowkbp4'] = dataset_constant['lowkbp']

base = ['i', 'indices', 'length', 'mask', 'anchor_index', 'dist']

feature_map = {
    'ann': base + ['entity_indices', 'entity_attention'],
    'cnn': base + ['prune_indices', 'prune_length', 'prune_mask', 'prune_footprint'],
    'lstm': base + ['prune_indices', 'prune_length', 'prune_mask', 'prune_footprint'],
    'gru': base + ['prune_indices', 'prune_length', 'prune_mask', 'prune_footprint'],
    'gcn': base + ['dep', 'prune_dep'],
    'bertgcn': base + ['cls_text_sep_indices',
                       'cls_text_sep_length',
                       'cls_text_sep_segment_ids',
                       'transform',
                       'dep',
                       'prune_dep'],

    'bertcnn': base + ['cls_text_sep_indices',
                       'cls_text_sep_length',
                       'cls_text_sep_segment_ids',
                       'transform',
                       'dep', 'prune_footprint',
                       'prune_dep'],
    'bertmlp': base + ['cls_text_sep_indices',
                       'cls_text_sep_length',
                       'cls_text_sep_segment_ids',
                       'transform',
                       'prune_footprint'],

    'berted': base + ['cls_text_sep_indices',
                      'cls_text_sep_length',
                      'cls_text_sep_segment_ids',
                      'transform'],
    'bertlinear': base + ['cls_text_sep_indices',
                          'cls_text_sep_length',
                          'cls_text_sep_segment_ids',
                          'transform',
                          'prune_indices', 'prune_length', 'prune_mask', 'prune_footprint'],
    'bertdm': base + ['cls_text_sep_indices',
                      'cls_text_sep_length',
                      'cls_text_sep_segment_ids',
                      'transform'],
    'mlp': base + ['emb']
}

encoder_class = {
    'ann': ANN,
    'cnn': CNN,
    'lstm': LSTM,
    'gru': GRU,
    'gcn': GCN,
    'bertgcn': BertGCN,
    'bertcnn': BertCNN,
    'bertmlp': BertMLP,
    'bertlinear': BertLinear,
    'mlp': MLP,

}

classificaion_class = {
    'cnn': CNNClassifier,
    'gcn': GCNClassifier,
    'berted': BertEDClassifier,
    'bertgcn': BertGCNClassifier,
    'bertdm': BertDMClassifier,

}

fsl_class = {
    'proto': PrototypicalNetwork,
    'attproto': AttPrototypicalNetwork,
    'relation': RelationNetwork,
    'matching': MatchingNetwork,
    'induction': InductionNetwork,
    'dmn': DProto,
    'melr': MELRNetwork,
    'melrplus': MELRPlus

}

meta_opt_net_class = {
    'svmcs': SVM_CS,
    'proto': Proto,
    'svmhe': SVM_He,
    'svmww': SVM_WW,
    'ridge': Ridge,
    'r2d2': R2D2
}
