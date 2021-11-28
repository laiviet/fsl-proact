import json
import os
from preprocess.utils import *


def distance_between_trigger_entity(path):
    files = [os.path.join(path, 'train.json'),
             os.path.join(path, 'dev.json'),
             os.path.join(path, 'test.json')]
    data = []
    for file in files:
        print('Load file: ', file)
        data += load_json(file)

    distances = []
    count = 0
    for item in data:
        ds = [item['trigger'][0] - arg[1] for arg in item['argument']]
        # ds = [abs(item['trigger'][0] - arg[0]) for arg in item['argument']]
        distances += ds
    #     if len(ds) == 1:
    #         if ds[0] > :
    #             count+=1
    # print(count)

    hist(distances)


def sentence_length(path):
    files = [os.path.join(path, x) for x in os.listdir(path) if x.endswith('.json')]
    data = []
    for file in files:
        print('Load file: ', file)
        data += load_json(file)

    lengths = [len(x['token']) for x in data]
    hist(lengths)


def argument_label(path):
    files = [os.path.join(path, x) for x in os.listdir(path) if x.endswith('.json')]
    data = []
    for file in files:
        print('Load file: ', file)
        data += load_json(file)

    argument_labels = []
    for item in data:
        # print(item['argument'])
        labels = [x[2] for x in item['argument']]
        argument_labels += labels
    hist(argument_labels)



def bert_lenght(path):
    files = [os.path.join(path, x) for x in os.listdir(path) if x.endswith('.bert-base-cased.json')]
    data = []
    for file in files:
        print('Load file: ', file)
        data += load_json(file)
    max_len = 0
    for item in data:
        bert = item['bert_indices']
        l =sum([len(x) for x in bert])
        if l > max_len:
            print(l)
            max_len=l

if __name__ == '__main__':
    # distance_between_trigger_entity('datasets/rams/fsl')
    # distance_between_trigger_entity('datasets/ace/fsl')

    bert_lenght('datasets/rams/fsl')
    bert_lenght('datasets/ace/fsl')

    # sentence_length('datasets/rams/fsl')
    # sentence_length('datasets/ace/fsl')
    #
    # argument_label('datasets/rams/fsl')
    # argument_label('datasets/ace/fsl')
    #
