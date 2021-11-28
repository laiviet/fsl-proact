import os
import json
import collections

TRAIN = ['train.jsonlines']
DEV = ['dev.jsonlines']
TEST = ['test.jsonlines']
ALL = TRAIN + DEV + TEST


def read_data(files):
    data = []
    for file in files:
        with open(file, 'r') as f:
            data += [json.loads(x) for x in f.readlines()]
    return data


def label_statistics(data):
    # 0 is by label
    # 1 is by number of instances
    SORT_BY = 0

    for k, v in data[0].items():
        print(k, v)

    triggers = []
    for item in data:
        triggers += item['evt_triggers']
    labels3 = [x[2][0][0] for x in triggers]
    labels1 = []
    labels2 = []

    for label in labels3:
        parts = label.split('.')
        labels1.append(parts[0])
        labels2.append(parts[0] + '.' + parts[1])

    counter1 = collections.Counter()
    counter2 = collections.Counter()
    counter3 = collections.Counter()
    counter1.update(labels1)
    counter2.update(labels2)
    counter3.update(labels3)

    print('-' * 80)
    level1 = [(k, v) for k, v in counter1.items()]
    level1 = sorted(level1, key=lambda x: x[SORT_BY])
    print(len(level1))
    for k, v in level1:
        print(k, v)
    print('-' * 80)
    level2 = [(k, v) for k, v in counter2.items()]
    level2 = sorted(level2, key=lambda x: x[SORT_BY])
    print(len(level2))

    for k, v in level2:
        print(k, v)
    print('-' * 80)
    level3 = [(k, v) for k, v in counter3.items()]
    level3 = sorted(level3, key=lambda x: x[SORT_BY])
    print(len(level3))
    for k, v in level3:
        print(k, v)


def trigger_length(data):
    triggers = []
    for item in data:
        triggers += item['evt_triggers']

    print(triggers[0])
    lengths = [x[1] - x[0] + 1 for x in triggers]

    counter = collections.Counter()
    counter.update(lengths)
    for k, v in counter.items():
        print(k, v)


if __name__ == '__main__':
    data = read_data(ALL)
    # label_statistics(data)
    trigger_length(data)
