import json
import collections
from preprocess.utils import *
import pandas as pd
import os

print(os.getcwd())

TRAIN = '../datasets/fed/train_data.csv'
DEV = '../datasets/fed/val_data.csv'
TEST = '../datasets/fed/test_data.csv'


# event = {
#     'id': '{}/{}'.format(doc_id, trigger_id),
#     'token': tokens,
#     'trigger': [trigger_span_start, trigger_span_end],
#     'label': label[0][0],
#     'argument': arguments
# }


def find_position(start, end, lengths):
    offset = 0
    for i, l in enumerate(lengths):
        if offset + l > start:
            return i, start - offset, end - offset
        offset += l
    print('Cannot resolve:')
    print(start)
    print(end)
    print(lengths)
    return 0, start, end


def read_data(files):
    events = []
    for file in files:
        df = pd.read_csv(file)

        for id, target_word, sentence, depparse, trigger_idx, sense_key, label in \
                zip(df.idx, df.target_word, df.sentence, df.depparse, df.target_idx, df.sense_key, df.label):
            event = {
                'id': id,
                'token': sentence.split(' '),
                'trigger': [trigger_idx, trigger_idx],
                'label': label,
                'argument': []
            }
            events.append(event)
    return events


def load_supervised():
    train = read_data([TRAIN])
    dev = read_data([DEV])
    test = read_data([TEST])

    label = {x['label'] for x in train + dev + test}

    print(len(label))
    return train, dev, test


def print_statistic(data):
    print('-' * 20)
    labels = [x['label'] for x in data]
    counter = collections.Counter()
    counter.update(labels)

    # print(len(counter))
    print(len(data))

    # for k, v in counter.items():
    #     print(k, v)


def load_fsl():
    all_data = read_data([TRAIN, DEV, TEST])
    train_label_set = 'abcdefghijklmn'
    dev_label_set = 'opqr'
    test_label_set = 'stuvwxyz'
    negative = 'O'

    print(len(all_data))

    counter = collections.Counter()
    counter.update([x['label'] for x in all_data])
    stats = [(k, v) for k, v in counter.items()]
    stats.sort(key=lambda x: x[1])
    seleted_labels = {k for k, v in stats if v >20}

    selected_data = [x for x in all_data if x['label'] in seleted_labels]
    negative = [x for x in all_data if x['label'] == 'Other']

    train = [x for x in selected_data if x['label'][0] in train_label_set]
    dev = [x for x in selected_data if x['label'][0] in dev_label_set]
    test = [x for x in selected_data if x['label'][0] in test_label_set]

    print(len(negative))
    train_neg = negative[:50000]
    dev_neg = negative[50000:70000]
    test_neg = negative[70000:]

    return train, dev, test, train_neg, dev_neg, test_neg


if __name__ == '__main__':
    # train, dev, test, train_neg, dev_neg, test_neg = load_fsl()
    # #
    # save_json(train, '../datasets/fed/fsl/train.json')
    # save_json(dev, '../datasets/fed/fsl/dev.json')
    # save_json(test, '../datasets/fed/fsl/test.json')
    # save_json(train_neg, '../datasets/fed/fsl/train.negative.json')
    # save_json(dev_neg, '../datasets/fed/fsl/dev.negative.json')
    # save_json(test_neg, '../datasets/fed/fsl/test.negative.json')

    train, dev, test = load_supervised()
    save_json(train, '../datasets/fed/supervised/train.json')
    save_json(dev, '../datasets/fed/supervised/dev.json')
    save_json(test, '../datasets/fed/supervised/test.json')
