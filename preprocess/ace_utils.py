import json
import collections
from preprocess.utils import *


def load_supervised():
    train = load_json('datasets/ace/original/train.json')
    dev = load_json('datasets/ace/original/valid.json')
    test = load_json('datasets/ace/original/test.json')

    return train, dev, test


def hist(populations, key=0):
    counter = collections.Counter()
    counter.update(populations)
    stats = [(k, v) for k, v in counter.items()]
    stats = sorted(stats, key=lambda x: x[key])
    for k, v in stats:
        print('{:4d}\t{}'.format(v, k))


def print_statistic(data):
    labels = [x['label'] for x in data]
    counter = collections.Counter()
    counter.update(labels)

    print('-' * 80)
    print('#class: ', len(set(labels)))
    print('#sample:', len(labels))

    hist(labels)

    sent_lengths = [len(x['token']) for x in data]

    print('| Max setence length: ', max(sent_lengths))
    # hist(sent_lengths)


def load_fsl():
    all_data = load_json('datasets/ace/original/train.json')
    all_data += load_json('datasets/ace/original/valid.json')
    all_data += load_json('datasets/ace/original/test.json')
    labels = [x['label'] for x in all_data]

    ignore_label_set = [
        'Justice.Acquit',
        'Justice.Extradite',
        'Justice.Pardon',
        'Personnel.Nominate'
    ]
    train_label_set = ['Business', 'Conflict', 'Contact', 'Justice']

    train = []
    dev = []
    test = []

    rr = True

    for item in all_data:
        if item['label'] in ignore_label_set:
            continue
        label_parts = item['label'].split('.')
        if label_parts[0] in train_label_set:
            train.append(item)
        else:
            if rr:
                dev.append(item)
                rr = not rr
            else:
                test.append(item)
                rr = not rr

    # print_statistic(train)
    # print_statistic(dev)
    # print_statistic(test)

    return train, dev, test


if __name__ == '__main__':
    # train, dev, test = load_fsl()
    #
    # save_json(train, 'datasets/ace/fsl/train.json')
    # save_json(dev, 'datasets/ace/fsl/dev.json')
    # save_json(test, 'datasets/ace/fsl/test.json')

    train, dev, test = load_supervised()
    save_json(train, 'datasets/ace/supervised/train.json')
    save_json(dev, 'datasets/ace/supervised/dev.json')
    save_json(test, 'datasets/ace/supervised/test.json')
