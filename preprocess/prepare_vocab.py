from preprocess.utils import *
import numpy
import pickle

glove = '../dataset/embedding/glove.6B.300d.txt'


def make_vocab():
    data = load_json('datasets/ace/fsl/train.json')
    data += load_json('datasets/ace/fsl/dev.json')
    data += load_json('datasets/ace/fsl/test.json')

    data += load_json('datasets/rams/fsl/train.json')
    data += load_json('datasets/rams/fsl/dev.json')
    data += load_json('datasets/rams/fsl/test.json')

    data += load_json('datasets/lowkbp0/fsl/train.json')
    data += load_json('datasets/lowkbp0/fsl/dev.json')
    data += load_json('datasets/lowkbp0/fsl/test.json')

    data += load_json('datasets/semcor/supervised/train.json')
    data += load_json('datasets/semcor/supervised/dev.json')
    vocab = set()
    for x in data:
        vocab.update([t.lower() for t in x['token']])

    vocab_map = {'<PAD>': 0,
                 '<UNK>': 1}

    with open(glove, 'r') as f:
        lines = f.readlines()
    dim = len(lines[0].split(' ')) - 1

    vectors = [
        [0.0 for _ in range(dim)],
        None
    ]
    i = 2
    for line in lines:
        parts = line.split(' ')
        word = parts[0]
        vec = [float(x) for x in parts[1:]]
        if word in vocab:
            vocab_map[word] = i
            vectors.append(vec)
            i += 1
    unk_vec = vectors[vocab_map['the']]
    vectors[1] = unk_vec
    vectors = numpy.array(vectors)
    with open('datasets/vocab.pkl', 'wb') as f:
        pickle.dump((vocab_map, vectors), f)
    print(len(vocab_map))


def make_label_mapping(dataset):
    data = load_json('datasets/{}/fsl/train.json'.format(dataset))
    data += load_json('datasets/{}/fsl/dev.json'.format(dataset))
    data += load_json('datasets/{}/fsl/test.json'.format(dataset))

    labels = {x['label'] for x in data}
    labels = sorted(labels)
    label_map = {l: i + 1 for i, l in enumerate(labels)}
    label_map['Other'] = 0

    labels = set()
    for x in data:
        labels.update([e[2] for e in x['argument']])
    labels = sorted(labels)
    argument_map = {l: i + 1 for i, l in enumerate(labels)}
    argument_map['Other'] = 0

    m = make_stanford_mapping()
    m['label'] = label_map
    m['argument'] = argument_map

    save_json(m, 'datasets/{}/fsl/label_map.json'.format(dataset))
    # save_json(m, 'datasets/{}/supervised/label_map.json'.format(dataset))


def make_label_mapping_semcor(dataset):
    data = load_json('datasets/{}/supervised/train.json'.format(dataset))
    data += load_json('datasets/{}/supervised/dev.json'.format(dataset))
    labels = set()
    for x in data:
        labels.update(x['candidate'])
    labels = sorted(labels)
    label_map = {l: i for i, l in enumerate(labels)}

    m = make_stanford_mapping()
    m['label'] = label_map

    save_json(m, 'datasets/{}/fsl/label_map.json'.format(dataset))
    save_json(m, 'datasets/{}/supervised/label_map.json'.format(dataset))


def make_stanford_mapping():
    data = load_json('datasets/ace/fsl/train.parse.json')
    data += load_json('datasets/ace/fsl/dev.parse.json')
    data += load_json('datasets/ace/fsl/test.parse.json')

    data += load_json('datasets/lowkbp0/fsl/train.parse.json')
    data += load_json('datasets/lowkbp0/fsl/dev.parse.json')
    data += load_json('datasets/lowkbp0/fsl/test.parse.json')

    data += load_json('datasets/rams/fsl/train.parse.json')
    data += load_json('datasets/rams/fsl/dev.parse.json')
    data += load_json('datasets/rams/fsl/test.parse.json')

    data += load_json('datasets/semcor/supervised/train.parse.json')
    data += load_json('datasets/semcor/supervised/dev.parse.json')

    ner_labels = set()
    pos_labels = set()
    dep_labels = set()
    for x in data:
        ner_labels.update(x['stanford_ner'])
        pos_labels.update(x['stanford_pos'])
        dep_labels.update([e[2] for e in x['edge']])

    _dep_labels = set()
    for dep in dep_labels:
        if ':' in dep:
            _dep_labels.add(dep.split(':')[0])
        else:
            _dep_labels.add(dep)

    ner_labels = sorted(ner_labels)
    pos_labels = sorted(pos_labels)
    dep_labels = sorted(_dep_labels)

    ner_map = {l: i for i, l in enumerate(ner_labels)}
    pos_map = {l: i for i, l in enumerate(pos_labels)}
    dep_map = {l: i for i, l in enumerate(dep_labels)}

    m = {
        'stanford_ner': ner_map,
        'stanford_pos': pos_map,
        'standord_dependency': dep_map
    }

    return m


if __name__ == '__main__':
    # make_vocab()
    # make_label_mapping('ace')
    # make_label_mapping('rams')
    # make_label_mapping('rams2')
    # make_label_mapping('fed')
    make_label_mapping('lowkbp0')
    make_label_mapping('lowkbp1')
    make_label_mapping('lowkbp2')
    make_label_mapping('lowkbp3')
    make_label_mapping('lowkbp4')
    # make_label_mapping_semcor('semcor')
