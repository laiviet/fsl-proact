from preprocess.utils import *
import os
import stanza
import re
import nltk
import collections
from os.path import join
import glob

from nltk import word_tokenize


def find_all(sentence, trigger):
    matched = []
    l = len(trigger)
    first = trigger[0]
    for i in range(len(sentence) - l + 1):
        if sentence[i] == first:
            if sentence[i:i + l] == trigger:
                matched.append(i)
    return matched


error = [0 for _ in range(10)]


def convert(item, label):
    sentence, trigger, index = item[:3]
    s, e, l = index
    tokens = sentence.split(' ')
    # if len(tokens) != l:
    #     error[3] += 1
    trigger = trigger.strip()

    trigger_tokens = trigger.split()
    # if len(trigger_tokens) > 3:
    #     return None
    if s == e == 0:
        # error[5]+=1
        return None

    start = s - 1
    end = l - e + 1
    if start >= end:
        end = start + len(trigger_tokens)

    trigger2 = ' '.join(tokens[start:end])
    # print(tokens[l - e:s + 1])

    if trigger != trigger2:
        if trigger2[:len(trigger)] != trigger:
            error[4] += 1
            print('-' * 40)
            print(sentence)
            print(s, e, l)
            print(trigger)
            print(trigger2)
            print(label)


def character_to_token(tokens, cix):
    """

    :param tokens: List of tokens
    :param cix: index of character
    :return:
    """
    offset = -1
    i = -1
    while True:
        i += 1
        offset += len(tokens[i]) + 1

        # print(offset, i)
        if offset > cix:
            return i


def test_character_to_token():
    sent = 'I will run this on the Low_Resource_KBP dataset (that appears in the Dynamic Memory Network paper).'
    tokens = sent.split(' ')
    assert character_to_token(tokens, 0) == 0, 'Got: {}'.format(character_to_token(tokens, 0))
    assert character_to_token(tokens, 2) == 1, 'Got: {}'.format(character_to_token(tokens, 2))
    assert character_to_token(tokens, 3) == 1, 'Got: {}'.format(character_to_token(tokens, 3))
    assert character_to_token(tokens, 4) == 1, 'Got: {}'.format(character_to_token(tokens, 4))
    assert character_to_token(tokens, 5) == 1, 'Got: {}'.format(character_to_token(tokens, 5))
    assert character_to_token(tokens, 7) == 2, 'Got: {}'.format(character_to_token(tokens, 6))


test_character_to_token()


def convert3(item, label):
    global error
    sentence, trigger, index = item[:3]
    s, e, l = index
    # tokens = sentence.split()
    # trigger_tokens = trigger.split()
    tokens = word_tokenize(sentence)
    trigger_tokens = word_tokenize(trigger)
    tokens = [x for x in tokens if len(x) > 0]
    trigger_tokens = [x for x in trigger_tokens if len(x) > 0]

    sentence = ' '.join(tokens)
    trigger = ' '.join(trigger_tokens)

    if s == e == 0:
        # error[0] += 1
        return None

    start = s - 1
    end = start + len(trigger_tokens)
    trigger2 = ' '.join(tokens[start:end])

    if trigger == trigger2 and len(trigger_tokens) < 4:
        return {
            'argument': [],
            'token': tokens,
            'trigger': [start, end - 1],
            'label': label,

        }

    if trigger2[:len(trigger)] == trigger and len(trigger_tokens) < 4:
        tokens = tokens[:start] + trigger_tokens + trigger2[len(trigger):].split() + tokens[end:]
        return {
            'argument': [],
            'token': tokens,
            'trigger': [start, end - 1],
            'label': label
        }
    matched = find_all(sentence, trigger)
    #
    if len(trigger_tokens) > 3:
        # error[1] += 1
        # print(40 * '-')
        # print(sentence)
        # print(trigger)
        # print(label)
        return None

    if len(matched) == 0:
        # print(40 * '-')
        # print(sentence)
        # print(trigger)
        # print(label)
        # error[2] += 1
        return None
    elif len(matched) > 1:
        # error[2] += 1
        prev = ' '.join(tokens[:s])
        l = len(prev)
        closet = 0
        current_distance = abs(matched[closet] - l)
        for i, position in enumerate(matched):
            if abs(position - l) < current_distance:
                closet = i
                current_distance = abs(position - l)
    else:
        closet = 0

    start = character_to_token(tokens, matched[closet])
    end = start + len(trigger_tokens)

    trigger_tokens2 = tokens[start:start + len(trigger_tokens)]
    trigger2 = ' '.join(trigger_tokens2)
    if trigger2[:len(trigger)] == trigger:
        tokens = tokens[:start] + trigger_tokens + trigger2[len(trigger):].split() + tokens[end:]

    return {
        'argument': [],
        'token': tokens,
        'trigger': [start, end - 1],
        'label': label
    }


def load_fsl():
    all_data = load_json('datasets/lowkbp/Few-Shot_ED.json')
    data = {}
    for label, l_data in all_data.items():
        for item in l_data:
            x = convert3(item, label)
            if x != None:
                identity = x['token'] + x['trigger']
                identity = tuple(identity)
                data[identity] = x

    data = [v for k, v in data.items()]

    counter = collections.Counter()

    dropped = [
        'Wine.Grape-Variety-Composition',
        'Music.Track-Contribution',
        'Projects.Project-Participation',
        'Film.Dubbing-Performance',
        'Conflict.Self-Immolation',
        'Justice.Acquit',
        'Justice.Pardon',
        'Business.Merge-Org'
    ]

    data = [x for x in data if x['label'] not in dropped]
    labels = [x['label'] for x in data]
    counter.update(labels)
    hist = sorted(counter.items(), key=lambda x: x[1])
    for k, v in hist:
        print(f'{v:5d} {k}')

    save_json(data, 'datasets/lowkbp/cleaned.json')

    categorized = {}
    for x in data:
        l = x['label']
        if x['label'] not in categorized:
            categorized[x['label']] = [x]
        elif len(categorized[l]) < 200:
            categorized[l].append(x)

    _data = []
    for k, v in categorized.items():
        _data += v
    save_json(_data, 'datasets/lowkbp/top200.json')


def split_five_fold():
    all_data = load_json('datasets/lowkbp/top200.json')

    for i in range(len(all_data)):
        all_data[i]['id'] = f'lowkbp#{i}'
    original_labels = {x['label'] for x in all_data}

    labels = sorted(original_labels)
    labels = labels + labels
    labels = labels[:100]

    for i in range(5):
        save_folder = f'datasets/lowkbp{i}/fsl'
        dev = set(labels[(i * 20): (i * 20 + 10)])
        test = set(labels[(i * 20 + 10): (i * 20 + 20)])
        train = original_labels.difference(dev.union(test))

        train_data = [x for x in all_data if x['label'] in train]
        dev_data = [x for x in all_data if x['label'] in dev]
        test_data = [x for x in all_data if x['label'] in test]

        os.makedirs(save_folder, exist_ok=True)

        save_json(train_data, join(save_folder, 'train.json'))
        save_json(dev_data, join(save_folder, 'dev.json'))
        save_json(test_data, join(save_folder, 'test.json'))

        print(dev.isdisjoint(test))
        print(dev.isdisjoint(train))
        print(test.isdisjoint(train))


def analyze_label():
    from itertools import combinations

    all_combs = combinations([i for i in range(5)], 5)

    labels = []
    for i in range(5):
        dev = load_json(f'datasets/lowkbp{i}/dev.json')
        test = load_json(f'datasets/lowkbp{i}/test.json')

        _labels = {x['label'] for x in dev + test}
        labels.append(_labels)

    for comb in all_combs:
        l = set()
        for i in comb:
            l.update(labels[i])
        print(comb, len(l))


def check_label_overlapping():
    for dataset in [f'lowkbp{i}' for i in range(5)]:
        print(40 * '-')
        train = load_json(f'datasets/{dataset}/train.json')
        dev = load_json(f'datasets/{dataset}/dev.json')
        test = load_json(f'datasets/{dataset}/test.json')

        train = {x['label'] for x in train}
        dev = {x['label'] for x in dev}
        test = {x['label'] for x in test}
        # print(40 * '-')
        # for x in sorted(train):
        #     print(x)
        # print(40 * '-')
        # for x in sorted(dev):
        #     print(x)
        # print(40 * '-')
        # for x in sorted(test):
        #     print(x)
        # print(40 * '-')

        print(len(train), len(dev), len(test))
        print(len(train.intersection(dev)))
        print(len(train.intersection(test)))
        print(len(test.intersection(dev)))


def histogram(population):
    from collections import Counter
    c = Counter()
    c.update(population)

    for k, v in c.items():
        print(v, k)


def check_data_population():
    for dataset in [f'lowkbp{i}' for i in range(5)]:
        print(40 * '-')
        train = load_json(f'datasets/{dataset}/fsl/train.json')
        dev = load_json(f'datasets/{dataset}/fsl/dev.json')
        test = load_json(f'datasets/{dataset}/fsl/test.json')
        train = [x['label'] for x in train]
        dev = [x['label'] for x in dev]
        test = [x['label'] for x in test]

        histogram(train)
        histogram(dev)
        histogram(test)

        print(40 * '-')


if __name__ == '__main__':
    # load_fsl()
    # split_five_fold()
    # analyze_label()
    # check_label_overlapping()
    check_data_population()