import corenlp
import random
from preprocess.rams_utils import *
import multiprocessing

# hostname = 'lengendary2.cs.uoregon.edu'
hostname = 'localhost'

ld1 = [corenlp.CoreNLPClient(annotators="pos ner depparse".split(),
                             start_server=False,
                             timeout=50000,
                             endpoint='http://legendary1.cs.uoregon.edu:{}'.format(port))
       for port in range(9000, 9010)]
ld2 = [corenlp.CoreNLPClient(annotators="pos ner depparse".split(),
                             start_server=False,
                             timeout=50000,
                             endpoint='http://legendary2.cs.uoregon.edu:{}'.format(port))
       for port in range(9000, 9010)]

# clients = ld1 + ld2
clients = ld2
properties = {
    'inputFormat': 'text',
    'outputFormat': clients[0].default_output_format,
    'serializer': 'edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer',
    'tokenize.whitespace': 'true',
    'tokenize.language': 'Whitespace',
    'ssplit.eolonly': 'true'
}


def annotate(sentence):
    client = random.choice(clients)
    doc = client.annotate(sentence, properties=properties)
    return doc


def parse_an_item(item):
    text = ' '.join(item['token'])

    doc = annotate(text)

    sentences = doc.sentence
    assert len(sentences) == 1, '{} sentences, check {}'.format(len(sentences), text)
    ners = []
    poses = []
    # print(sentences[0])
    for word in sentences[0].token:
        ners.append(word.ner)
        poses.append(word.pos)
    tree = sentences[0].enhancedDependencies
    edges = []
    for edge in tree.edge:
        src = edge.source - 1
        tgt = edge.target - 1
        relation = edge.dep
        edges.append([src, tgt, relation])
        assert src > -1, "Negative source"
        assert tgt > -1, 'Negative target'
    return_item = {
        'id': item['id'],
        'stanford_ner': ners,
        'stanford_pos': poses,
        'edge': edges
    }
    return return_item


def parse(prefix):
    pool = multiprocessing.Pool(10)
    data = load_json('{}.json'.format(prefix))

    result = pool.map(parse_an_item, data)
    print(len(data), len(result))

    with open('{}.parse.json'.format(prefix), 'w') as f:
        json.dump(result, f)


def test():
    item = {
        "argument": [
            [
                "21",
                "21",
                "Vehicle"
            ],
            [
                "26",
                "26",
                "Artifact"
            ],
            [
                "30",
                "30",
                "Destination"
            ]
        ],
        "id": "bc/timex2norm/CNN_CF_20030303.1900.00#2",
        "label": "Movement.Transport",
        "token": [
            "even",
            "as",
            "the",
            "secretary",
            "of",
            "homeland",
            "security",
            "was",
            "putting",
            "his",
            "people",
            "on",
            "high",
            "alert",
            "last",
            "month",
            ",",
            "a",
            "30-foot",
            "Cuban",
            "patrol",
            "boat",
            "with",
            "four",
            "heavily",
            "armed",
            "men",
            "landed",
            "on",
            "American",
            "shores",
            ",",
            "utterly",
            "undetected",
            "by",
            "the",
            "Coast",
            "Guard",
            "Secretary",
            "Ridge",
            "now",
            "leads",
            "."
        ],
        "trigger": [
            27,
            27
        ]
    }

    print(parse(item))


if __name__ == '__main__':
    # parse('datasets/ace/fsl/train')
    # parse('datasets/ace/fsl/dev')
    # parse('datasets/ace/fsl/test')

    # parse('datasets/ace/fsl/train.negative')
    # parse('datasets/ace/fsl/dev.negative')
    # parse('datasets/ace/fsl/test.negative')

    # parse('datasets/ace/supervised/train')
    # parse('datasets/ace/supervised/dev')
    # parse('datasets/ace/supervised/test')
    #
    # parse('datasets/ace/supervised/train.negative')
    # parse('datasets/ace/supervised/dev.negative')
    # parse('datasets/ace/supervised/test.negative')

    # parse('datasets/rams/fsl/train')
    # parse('datasets/rams/fsl/dev')
    # parse('datasets/rams/fsl/test')

    # parse('datasets/rams/fsl/train.negative')
    # parse('datasets/rams/fsl/dev.negative')
    # parse('datasets/rams/fsl/test.negative')

    # parse('datasets/rams/supervised/train')
    # parse('datasets/rams/supervised/dev')
    # parse('datasets/rams/supervised/test')
    #
    # parse('datasets/rams/supervised/train.negative')
    # parse('datasets/rams/supervised/dev.negative')
    # parse('datasets/rams/supervised/test.negative')

    # parse('datasets/semcor/supervised/dev')
    # parse('datasets/semcor/supervised/train')

    # parse('datasets/cyber/train')
    # parse('datasets/cyber/dev')
    # parse('datasets/cyber/test')

    # parse('datasets/fed/fsl/train')
    # parse('datasets/fed/fsl/dev')
    # parse('datasets/fed/fsl/test')
    #
    # parse('datasets/fed/fsl/train.negative')
    # parse('datasets/fed/fsl/dev.negative')
    # parse('datasets/fed/fsl/test.negative')
    #
    # parse('datasets/fed/supervised/train')
    # parse('datasets/fed/supervised/dev')
    # parse('datasets/fed/supervised/test')
    #
    # parse('datasets/fed/supervised/train.negative')
    # parse('datasets/fed/supervised/dev.negative')
    # parse('datasets/fed/supervised/test.negative')

    parse(f'datasets/lowkbp0/fsl/dev')
    parse(f'datasets/lowkbp1/fsl/dev')
    parse(f'datasets/lowkbp2/fsl/dev')
    parse(f'datasets/lowkbp3/fsl/dev')
    parse(f'datasets/lowkbp4/fsl/dev')
    parse(f'datasets/lowkbp0/fsl/test')
    parse(f'datasets/lowkbp1/fsl/test')
    parse(f'datasets/lowkbp2/fsl/test')
    parse(f'datasets/lowkbp3/fsl/test')
    parse(f'datasets/lowkbp4/fsl/test')
    parse(f'datasets/lowkbp0/fsl/train')
    parse(f'datasets/lowkbp1/fsl/train')
    parse(f'datasets/lowkbp2/fsl/train')
    parse(f'datasets/lowkbp3/fsl/train')
    parse(f'datasets/lowkbp4/fsl/train')
