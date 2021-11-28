from preprocess.utils import *
import random
import copy


def generate_negative_fsl(prefix):
    SELECTED_POS = ['JJ', 'NN', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    raw = load_json('{}.json'.format(prefix))
    parsed = load_json('{}.parse.json'.format(prefix))

    all_sent_ids = {item['id'] for item in raw}

    negative_raw_examples = []
    negative_parse_examples = []

    for id in all_sent_ids:
        raw_items = [x for x in raw if x['id'] == id]
        parsed_items = [x for x in parsed if x['id'] == id]

        assert len(raw_items) == len(parsed_items)

        ignore_ids = set()
        for item in raw_items:
            s, t = item['trigger']
            for i in range(s, t + 1):
                ignore_ids.add(i)
            for s, t, l in item['argument']:
                for i in range(s, t + 1):
                    ignore_ids.add(i)
        # Get POS
        pos = parsed_items[0]['stanford_pos']
        available_indices = []
        for i, p in enumerate(pos):
            if i in ignore_ids:
                continue
            if p in SELECTED_POS:
                available_indices.append(i)

        # Select
        if len(available_indices) > 0:
            trigger_index = random.choice(available_indices)
            new_raw_item = copy.deepcopy(raw_items[0])
            new_raw_item['trigger'] = [trigger_index, trigger_index]
            new_raw_item['label'] = 'Other'

            negative_raw_examples.append(new_raw_item)
            negative_parse_examples.append(copy.deepcopy(parsed_items[0]))

    for r, g in zip(negative_raw_examples, negative_parse_examples):
        assert r['id'] == g['id']

    print('Negative: ', len(negative_raw_examples))
    save_json(negative_raw_examples, '{}.negative.json'.format(prefix))
    save_json(negative_parse_examples, '{}.negative.parse.json'.format(prefix))


def generate_negative_supervised(prefix):
    SELECTED_POS = ['JJ', 'NN', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    raw = load_json('{}.json'.format(prefix))
    parsed = load_json('{}.parse.json'.format(prefix))

    all_sent_ids = {item['id'] for item in raw}

    negative_raw_examples = []
    negative_parse_examples = []

    for id in all_sent_ids:
        raw_items = [x for x in raw if x['id'] == id]
        parsed_items = [x for x in parsed if x['id'] == id]

        assert len(raw_items) == len(parsed_items)

        ignore_ids = set()
        for item in raw_items:
            s, t = item['trigger']
            for i in range(s, t + 1):
                ignore_ids.add(i)
            for s, t, l in item['argument']:
                for i in range(s, t + 1):
                    ignore_ids.add(i)
        # Get POS
        pos = parsed_items[0]['stanford_pos']
        available_indices = []
        for i, p in enumerate(pos):
            if i in ignore_ids:
                continue
            # if p in SELECTED_POS:
            available_indices.append(i)

        # Select
        for trigger_index in available_indices:
            new_raw_item = copy.deepcopy(raw_items[0])
            new_raw_item['trigger'] = [trigger_index, trigger_index]
            new_raw_item['label'] = 'Other'

            negative_raw_examples.append(new_raw_item)
            negative_parse_examples.append(copy.deepcopy(parsed_items[0]))

    for r, g in zip(negative_raw_examples, negative_parse_examples):
        assert r['id'] == g['id']

    print('Negative: ', len(negative_raw_examples))
    save_json(negative_raw_examples, '{}.negative.json'.format(prefix))
    save_json(negative_parse_examples, '{}.negative.parse.json'.format(prefix))


if __name__ == '__main__':
    random.seed(1234)
    # generate_negative_fsl('datasets/ace/fsl/train')
    # generate_negative_fsl('datasets/ace/fsl/dev')
    # generate_negative_fsl('datasets/ace/fsl/test')
    # #
    # generate_negative_fsl('datasets/rams/fsl/train')
    # generate_negative_fsl('datasets/rams/fsl/dev')
    # generate_negative_fsl('datasets/rams/fsl/test')

    generate_negative_supervised('datasets/ace/supervised/train')
    generate_negative_supervised('datasets/ace/supervised/dev')
    generate_negative_supervised('datasets/ace/supervised/test')

    # generate_negative_supervised('datasets/rams/supervised/train')
    # generate_negative_supervised('datasets/rams/supervised/dev')
    # generate_negative_supervised('datasets/rams/supervised/test')
