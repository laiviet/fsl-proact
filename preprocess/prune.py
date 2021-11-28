import multiprocessing
from preprocess.utils import *

ML = 40


def bfs(id, adj, origin, destination, max_len=50):
    # print('Find: ', origin, destination)
    queue = [(origin, None)]
    trace = [(-1, None) for _ in range(max_len)]
    visited = set()

    while len(queue) > 0:
        visit, _ = queue.pop(0)
        visited.add(visit)
        # print('Visit: ', visit)
        if visit == destination:
            break
        if visit in adj:
            for n, l in adj[visit]:
                if n not in visited:
                    # print('Add:', n)
                    queue.append((n, l))
                    trace[n] = (visit, l)
    sub_tree = []
    current = destination
    try:
        while current != origin:
            n, l = trace[current]
            sub_tree.append([current, n, l])
            current = n
            if current == -1:
                return []
            # assert current > -1, 'id={} current ={}'.format(id, current)
    except:
        print(current)
    return sub_tree


def subtree(id, edges, trigger, arguments, neighbor=1):
    # print(edges)
    # print(trigger)
    # print(arguments)
    # Create adjunt map
    adj = {}
    for i, j, l in edges:
        if i not in adj:
            adj[i] = set([(j, l)])
        else:
            adj[i].add((j, l))
        if j not in adj:
            adj[j] = set([(i, l)])
        else:
            adj[j].add((i, l))

    pruned = set()
    paths = []

    # Find path within trigger
    start, end = trigger
    for i in range(start, end + 1):
        path = bfs(id, adj, start, i)
        paths.append(path)

    for start, end, _ in arguments:
        # find path between trigger and  argument
        paths.append(bfs(id, adj, trigger[0], start))
        # Find path within argument
        for i in range(start, end + 1):
            path = bfs(id, adj, start, i)
            paths.append(path)
    # print('--------')
    for path in paths:
        # print(path)
        for h, t, l in path:
            if [h, t, l] in edges:
                pruned.add((h, t, l))
            elif [t, h, l] in edges:
                pruned.add((t, h, l))
            else:
                print("Cannot find: ", h, t, l)
                exit(0)

    pruned = [list(x) for x in pruned]
    return pruned


def test_subtree():
    dataset, setting, corpus = 'rams', 'fsl', 'dev'
    raw = load_json('datasets/{}/{}/{}.json'.format(dataset, setting, corpus))
    graph = load_json('datasets/{}/{}/{}.parse.json'.format(dataset, setting, corpus))

    d = raw[0]
    g = graph[0]

    # print(' '.join(d['token']))
    pruned = subtree(g['edge'], d['trigger'], d['argument'])

    # print(pruned)


def prune_by_sentence_length_semcor(data):
    item, graph = data

    print('{} vs {}'.format(item['id'], graph['id']))

    assert item['id'] == graph['id'], '{} vs {}'.format(item['id'], graph['id'])
    k = ML // 2
    l = len(item['token'])
    t = item['trigger'][0]
    if l < ML:
        start, end = 0, l
    else:
        start = max(0, t - k)
        end = min(l, t + k)
    assert 0 <= start < l, "wrong start"
    assert 0 <= end <= l, "wrong end 0 <= {} < {}".format(end, l)
    assert 0 < end - start <= ML, 'Wrong start and end 0 < {} <= {}'.format(start - end, ML)

    # Crop tree
    crop = []
    for h, t, l in graph['edge']:
        if start <= h < end and start <= t < end:
            crop.append([h - start, t - start, l])

    trigger = item['trigger']
    new_trigger = [trigger[0] - start, trigger[1] - start]
    new_arguments = []
    pruned = subtree(item['id'], crop, new_trigger, new_arguments)

    crop_item = {
        'id': item['id'],
        'label': item['label'],
        'token': item['token'][start:end],
        'trigger_text': item['token'][start:end][new_trigger[0]:new_trigger[1] + 1],
        'trigger': new_trigger,
        'argument': new_arguments,
        'stanford_pos': graph['stanford_pos'][start:end],
        'stanford_ner': graph['stanford_ner'][start:end],
        'edge': crop,
        'edge_prune': pruned
    }
    return crop_item


def prune_by_sentence_length(data):
    item, graph = data

    # print('{} vs {}'.format(item['id'], graph['id']))

    assert item['id'] == graph['id'], '{} vs {}'.format(item['id'], graph['id'])
    k = ML // 2
    l = len(item['token'])
    t = item['trigger'][0]
    if l < ML:
        start, end = 0, l
    else:
        start = max(0, t - k)
        end = min(l, t + k)
    assert 0 <= start < l, "wrong start"
    assert 0 <= end <= l, "wrong end 0 <= {} < {}".format(end, l)
    assert 0 < end - start <= ML, 'Wrong start and end 0 < {} <= {}'.format(start - end, ML)

    # New trigger and argument
    trigger = item['trigger']
    new_trigger = [trigger[0] - start, trigger[1] - start]
    new_arguments = []
    for arg in item['argument']:
        if arg[0] >= start and arg[1] <= end:
            new_arguments.append([arg[0] - start, arg[1] - start, arg[2]])

    # Crop tree
    crop = []
    for h, t, l in graph['edge']:
        if start <= h < end and start <= t < end:
            crop.append([h - start, t - start, l])

    pruned = subtree(item['id'], crop, new_trigger, new_arguments)

    crop_item = {
        'id': item['id'],
        'label': item['label'],
        'token': item['token'][start:end],
        'trigger_text': item['token'][start:end][new_trigger[0]:new_trigger[1] + 1],
        'trigger': new_trigger,
        'argument': new_arguments,
        'stanford_pos': graph['stanford_pos'][start:end],
        'stanford_ner': graph['stanford_ner'][start:end],
        'edge': crop,
        'edge_prune': pruned
    }
    return crop_item


def prune(prefix):
    n_process = 20
    pool = multiprocessing.Pool(n_process)

    raw = load_json('{}.json'.format(prefix))
    graph = load_json('{}.parse.json'.format(prefix))

    # Single process
    # data = []
    # for x in zip(raw, graph):
    #     r,g = x
    #     print(r['id'], g['id'])
    #     data.append(prune_by_sentence_length(x))

    if 'semcor' in prefix:
        data = pool.map(prune_by_sentence_length_semcor, zip(raw, graph))
    else:
        data = pool.map(prune_by_sentence_length, zip(raw, graph))
    save_json(data, '{}.prune.json'.format(prefix))
    pool.close()


if __name__ == '__main__':
    # prune('datasets/ace/fsl/train')
    # prune('datasets/ace/fsl/dev')
    # prune('datasets/ace/fsl/test')

    # prune('datasets/ace/fsl/train.negative')
    # prune('datasets/ace/fsl/dev.negative')
    # prune('datasets/ace/fsl/test.negative')

    # prune('datasets/ace/supervised/train')
    # prune('datasets/ace/supervised/dev')
    # prune('datasets/ace/supervised/test')

    # prune('datasets/ace/supervised/train.negative')
    # prune('datasets/ace/supervised/dev.negative')
    # prune('datasets/ace/supervised/test.negative')

    # prune('datasets/rams/fsl/train')
    # prune('datasets/rams/fsl/dev')
    # prune('datasets/rams/fsl/test')

    # prune('datasets/rams/fsl/train.negative')
    # prune('datasets/rams/fsl/dev.negative')
    # prune('datasets/rams/fsl/test.negative')

    # prune('datasets/rams/supervised/train')
    # prune('datasets/rams/supervised/dev')
    # prune('datasets/rams/supervised/test')
    #
    # prune('datasets/rams/supervised/train.negative')
    # prune('datasets/rams/supervised/dev.negative')
    # prune('datasets/rams/supervised/test.negative')
    #
    # prune('datasets/semcor/supervised/train')
    # prune('datasets/semcor/supervised/dev')

    # prune('datasets/cyber/train')
    # prune('datasets/cyber/dev')
    # prune('datasets/cyber/test')

    # prune('datasets/fed/fsl/train')
    # prune('datasets/fed/fsl/dev')
    # prune('datasets/fed/fsl/test')
    #
    # prune('datasets/fed/fsl/train.negative')
    # prune('datasets/fed/fsl/dev.negative')
    # prune('datasets/fed/fsl/test.negative')
    #
    # prune('datasets/fed/supervised/train')
    # prune('datasets/fed/supervised/dev')
    # prune('datasets/fed/supervised/test')
    #
    # prune('datasets/fed/supervised/train.negative')
    # prune('datasets/fed/supervised/dev.negative')
    # prune('datasets/fed/supervised/test.negative')

    prune(f'datasets/lowkbp0/fsl/dev')
    prune(f'datasets/lowkbp1/fsl/dev')
    prune(f'datasets/lowkbp2/fsl/dev')
    prune(f'datasets/lowkbp3/fsl/dev')
    prune(f'datasets/lowkbp4/fsl/dev')
    prune(f'datasets/lowkbp0/fsl/test')
    prune(f'datasets/lowkbp1/fsl/test')
    prune(f'datasets/lowkbp2/fsl/test')
    prune(f'datasets/lowkbp3/fsl/test')
    prune(f'datasets/lowkbp4/fsl/test')
    prune(f'datasets/lowkbp0/fsl/train')
    prune(f'datasets/lowkbp1/fsl/train')
    prune(f'datasets/lowkbp2/fsl/train')
    prune(f'datasets/lowkbp3/fsl/train')
    prune(f'datasets/lowkbp4/fsl/train')
