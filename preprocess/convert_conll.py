import os
import json
import collections
import multiprocessing





def to_conll(item):
    remove_non_event_sentence = True
    bio = True

    sentences = item['sentences']
    triggers = item['evt_triggers']
    all_tokens = []
    all_labels = []
    lengths = []

    for sent in sentences:
        lengths.append(len(sent))
        all_tokens += sent
        all_labels += ['O' for _ in range(len(sent))]
    assert len(all_tokens) == len(all_labels)

    if bio:
        for start, end, label in triggers:
            label = label[0][0]
            all_labels[start] = 'B-' + label
            for i in range(start + 1, end + 1):
                all_labels[i] = 'I-' + label
    else:
        for start, end, label in triggers:
            label = label[0][0]
            all_labels[start] = label

    offset = 0
    sentence_tokens = []
    sentence_labels = []
    for i, l in enumerate(lengths):
        tokens = all_tokens[offset:offset + l]
        labels = all_labels[offset:offset + l]

        if remove_non_event_sentence:
            if not all([x == 'O' for x in labels]):
                sentence_tokens.append(tokens)
                sentence_labels.append(labels)
        else:
            sentence_tokens.append(tokens)
            sentence_labels.append(labels)
        offset += l

    return sentence_tokens, sentence_labels


def convert_to_conll(data):
    pool = multiprocessing.Pool(8)
    data = pool.map(to_conll, data)

    all_lines = []
    all_sentences = []

    for sentence_tokens, sentence_labels in data:
        for tokens, labels in zip(sentence_tokens, sentence_labels):
            lines = ['{}\t{}\n'.format(t, l) for t, l in zip(tokens, labels)]
            line = '\n'.join(lines)
            all_lines.append(line)
            sentence = ' '.join(tokens)
            all_sentences.append(sentence)
            # all_lines += ['\n']
    print(len(all_lines), len(set(all_lines)))
    print(len(all_sentences), len(set(all_sentences)))

    # with open('event-only.bio.conll', 'w') as f:
    #     f.writelines(all_lines)


if __name__ == '__main__':
    data = read_data(ALL)
    convert_to_conll(data)
