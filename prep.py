import argparse
import preprocess.rams_utils as rams_utils
import preprocess.tokenizer as tokenizer
import preprocess.ace_utils as ace_utils
import preprocess.graph as graph
import json
import multiprocessing
from preprocess.utils import *


def str_to_list(str):
    return str.split(',')


def run_rams_utils(args):
    if 'rams' in args.dataset:
        if 'fsl' in args.setting:
            train, dev, test = rams_utils.load_fsl()
            with open('datasets/rams/fsl/train.json', 'w') as f:
                json.dump(train, f)
            with open('datasets/rams/fsl/dev.json', 'w') as f:
                json.dump(dev, f)
            with open('datasets/rams/fsl/test.json', 'w') as f:
                json.dump(test, f)
        if 'supervised' in args.setting:
            train, dev, test = rams_utils.load_supervised()
            with open('datasets/rams/supervised/train.json', 'w') as f:
                json.dump(train, f)
            with open('datasets/rams/supervised/dev.json', 'w') as f:
                json.dump(dev, f)
            with open('datasets/rams/supervised/test.json', 'w') as f:
                json.dump(test, f)


def run_ace_utils(args):
    if 'ace' in args.dataset:
        if 'fsl' in args.setting:
            train, dev, test = ace_utils.load_fsl()
            with open('datasets/ace/fsl/train.json', 'w') as f:
                json.dump(train, f)
            with open('datasets/ace/fsl/dev.json', 'w') as f:
                json.dump(dev, f)
            with open('datasets/ace/fsl/test.json', 'w') as f:
                json.dump(test, f)
        if 'supervised' in args.setting:
            train, dev, test = ace_utils.load_supervised()
            with open('datasets/ace/supervised/train.json', 'w') as f:
                json.dump(train, f)
            with open('datasets/ace/supervised/dev.json', 'w') as f:
                json.dump(dev, f)
            with open('datasets/ace/supervised/test.json', 'w') as f:
                json.dump(test, f)


def run_graph(datasets, settings, corpus=('train', 'dev', 'test')):
    pool = multiprocessing.Pool(20)
    for d in datasets:
        for s in settings:
            for c in corpus:
                data = graph.load_json('datasets/{}/{}/{}.json'.format(d, s, c))
                parsed_data = pool.map(graph.parse, data)
                save_json(parsed_data, 'datasets/{}/{}/{}.parse.json'.format(d, s, c))
    pool.close()


def run_prune(datasets, settings, corpus=('train', 'dev', 'test')):
    pool = multiprocessing.Pool(20)
    for d in datasets:
        for s in settings:
            for c in corpus:
                data = graph.load_json('datasets/{}/{}/{}.json'.format(d, s, c))
                parsed_data = pool.map(graph.parse, data)
                save_json(parsed_data, 'datasets/{}/{}/{}.parse.json'.format(d, s, c))
    pool.close()


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str_to_list, default='ace,rams,semcor')
parser.add_argument('-s', '--setting', type=str_to_list, default='fsl')
parser.add_argument('-u', '--util', default=False, action='store_true')
parser.add_argument('-g', '--graph', default=False, action='store_true')
parser.add_argument('-p', '--prune', default=False, action='store_true')
parser.add_argument('-t', '--tokenize', default=False, action='store_true')

args = parser.parse_args()
if __name__ == '__main__':
    # Dataset format, corpus split
    if args.util:
        run_rams_utils(args)
        run_ace_utils(args)

    # POS, NER, Dependency Parsing
    if args.graph:
        run_graph(args.dataset, args.setting)

    if args.prune:
        run_prune(args.dataset, args.setting)

    # Tokenize
    if args.tokenize:
        tokenizer.tokenize_from_json_file(args.dataset, args.setting)
