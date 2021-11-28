from preprocess.utils import *
import os
import collections
import multiprocessing


# def read_all_eventive_sense():
#     def convert(x):
#         return '.'.join(x.split('.')[:2])
#
#     with open('datasets/semcor/deri_forms.txt') as f:
#         deri = f.read().split('\n')
#     with open('datasets/semcor/event_nouns.txt') as f:
#         noun = f.read().split('\n')
#
#     eventive_sense = {convert(x) for x in deri + noun if len(x) > 0}
#     print('Number of eventive sense: ', len(eventive_sense))
#
#     return eventive_sense


def read_semcor_dataset(path):
    # all_eventive_senses = read_all_eventive_sense()
    # print(list(all_eventive_senses)[:10])

    ignore_list = load_json('datasets/semcor/ignore.json')['label']

    with open(path, 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        parts = line.split('\t')
        sense = parts[1]
        id = parts[0]
        document = parts[2]
        anchor_index = int(parts[3])
        label = parts[4]
        if label in ignore_list:
            continue
        candidates = parts[5].split(';')

        if '%' in parts[5]:
            item = {
                'id': id,
                'doc': document,
                'anchor_index': anchor_index,
                'label': label,
                'candidate': candidates
            }
            data.append(item)
    return data


def filter_with_tagset(data, tagset):
    filtered = []
    for x in data:
        if x['label'] not in tagset:
            continue
        candidates = []
        for c in x['candidate']:
            if c in tagset:
                candidates.append(c)
        if len(candidates) > 1:
            y = x
            y['candidate'] = candidates
            filtered.append(y)
    print(len(data))
    print(len(filtered))
    return filtered


def crop_sentence(item):
    doc = item['doc'].split(' ')
    del item['doc']
    anchor_index = item['anchor_index']
    del item['anchor_index']

    punct = '?!.;'

    start = 0
    for i in range(0, anchor_index):
        if doc[i] in punct:
            start = i + 1
    end = len(doc)
    for i in range(len(doc) - 1, anchor_index, -1):
        if doc[i] in punct:
            end = i + 1

    tokens = doc[start:end]
    anchor_index = anchor_index - start
    item['token'] = tokens
    item['trigger'] = [anchor_index, anchor_index]

    assert 0 <= anchor_index < len(tokens)
    return item


def preprocess_semcor():
    data = read_semcor_dataset('../dataset/semcor/train.dat')
    candidates = []
    labels = []
    # counter = collections.Counter()
    for x in data:
        candidates += x['candidate']
        labels.append(x['label'])

    stats = hist(candidates, key=1, min=20, print_stat=False)

    sense_tagset = [x[0] for x in stats]

    sense_map = {k: i for i, k in enumerate(sense_tagset)}
    label_maps = {
        "label": sense_map
    }
    save_json(label_maps, 'datasets/semcor/supervised/label_map.json')

    filtered = filter_with_tagset(data, sense_tagset)
    p = multiprocessing.Pool(32)
    crop = p.map(crop_sentence, filtered)

    save_json(crop, 'datasets/semcor/supervised/all.crop.json')

    l = int(len(crop) * 0.9)
    train = crop[:l]
    dev = crop[l:]

    save_json(train, 'datasets/semcor/supervised/train.json')
    save_json(dev, 'datasets/semcor/supervised/dev.json')


if __name__ == '__main__':
    # read_all_eventive_sense()
    preprocess_semcor()
