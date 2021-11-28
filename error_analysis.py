from preprocess.utils import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def metrics(targets, predictions):
    labels = [x for x in range(1, 6)]
    p = 100 * precision_score(targets, predictions, labels=labels, average='micro')
    r = 100 * recall_score(targets, predictions, labels=labels, average='micro')
    f = 100 * f1_score(targets, predictions, labels=labels, average='micro')

    return p, r, f


def load_files(part):
    best_f = 0.5
    best_i = 0
    best_y = None
    for i in range(400, 6000, 400):
        # x = load_json('checkpoints/{}.dev.{}.json'.format(part,i))
        y = load_json('checkpoints/{}.test.{}.json'.format(part, i))
        target = np.stack(y['target']).reshape(-1)
        prediction = np.stack(y['prediction']).reshape(-1)
        p, r, f = metrics(target, prediction)
        if f > best_f:
            best_f = f
            best_y = y
            best_i = i

    print('Best at', best_i)
    target = np.stack(best_y['target']).reshape(-1)
    prediction = np.stack(best_y['prediction']).reshape(-1)

    p, r, f = metrics(target, prediction)
    print(f)
    return best_y


class Analyze():

    def __init__(self, base, full):
        self.raw = load_json('datasets/rams/fsl/test.prune.json')
        self.raw += load_json('datasets/rams/fsl/test.negative.prune.json')

        self.indices = load_json('datasets/rams/fsl/test.bert-base-cased.json')
        self.indices += load_json('datasets/rams/fsl/test.negative.bert-base-cased.json')

        self.base_pred = np.stack(base['prediction']).reshape(-1)
        self.full_pred = np.stack(full['prediction']).reshape(-1)
        self.target = np.stack(base['target']).reshape(-1)

        base_sample = np.stack(base['sample']).reshape(125, 4, 6, 9)
        self.sample = base_sample[:, :, :, 5:].reshape(-1)

    def analyze(self):
        self.show_performance()
        self.plot_accuracy_length()

    def show_performance(self):
        print('-------- Performance --------------------------')
        print('base', metrics(self.target, self.base_pred))
        print('full', metrics(self.target, self.full_pred))

    def extract_length_by_bin(self, pred, start, end):
        avg = lambda x: 100.0 * sum(x) / (len(x) + 1e-10)
        ml = 41
        f = lambda x, y: int(x == y)
        acc_by_length = [[] for _ in range(0, ml)]
        acc_by_prune_length = [[] for _ in range(0, ml)]
        f1_by_no_prune = [[] for _ in range(0, ml)]
        acc_by_sub_percentage = [[] for _ in range(0, 21)]
        # print(self.raw[0].keys())'
        lengths = []

        for p, t, s in zip(pred, self.target, self.sample):
            score = float(p == t)
            item = self.indices[s]
            l = len(item['glove_indices'])
            assert l < 41
            # if l <10:
            # if 30<=l<40:
            if start <= l < end:

                lengths.append(l)
                indices_idx = set()
                for h, t, label in self.raw[s]['edge_prune']:
                    indices_idx.update([h, t])
                indices_idx = sorted(indices_idx)
                pl = len(indices_idx)
                no_p = l - pl

                sub_percentage = int(100 * float(no_p) / float(l)) // 20
                # print(sl, l, sub_percentage)

                acc_by_length[l].append(score)
                # acc_by_prune_length[pl].append(score)
                f1_by_no_prune[no_p].append(score)
                acc_by_sub_percentage[sub_percentage].append(score)

        bins = [i for i in range(start, end)]
        acc_by_length = [avg(x) for x in acc_by_length[start:end]]
        # acc_by_prune_length = [avg(x) for x in acc_by_prune_length]
        f1_by_no_prune = [avg(x) for x in f1_by_no_prune]
        acc_by_sub_percentage = [avg(x) for x in acc_by_sub_percentage]

        # bins = self.bin(bins[1:-1], function='max')
        # acc_by_length = self.bin(acc_by_length)
        # f1_by_no_prune = self.bin(f1_by_no_prune)
        # acc_by_sub_length = self.bin(acc_by_sub_length)

        # print(acc_by_sub_percentage)

        return lengths, bins, acc_by_length, None, f1_by_no_prune, acc_by_sub_percentage

    def bin(self, data, bin=2, function='avg'):
        result = []
        for i in range(0, len(data), bin):
            if function == 'max':
                result.append(max(data[i:i + bin]))
            else:
                result.append(sum(data[i:i + bin]) / len(data[i:i + bin]))
        return result

    def plot_accuracy_length(self):
        start, end = 1, 41

        # start, end = 1, 10
        # plt.suptitle('Short sentences (1<=length<10)')
        #
        # start, end = 10, 20
        # plt.suptitle('Short sentences (10<=length<20)')
        #
        # start, end = 20, 30
        # plt.suptitle('Medium sentences (20<=length<30)')
        #
        # start, end = 30, 40
        # plt.suptitle('Long sentences (30<=length<40)')
        #
        # start, end = 40, 41
        # plt.suptitle('Super long sentences (40<=length)')

        lengths, x, by1, by2, by3, by4 = self.extract_length_by_bin(self.base_pred, start, end)
        lengths, x, fy1, fy2, fy3, fy4 = self.extract_length_by_bin(self.full_pred, start, end)

        plt.figure(figsize=(6, 10),dpi=300)

        plt.subplot(311)
        plt.title('Sentence length histogram')
        lengths = [x for x in lengths if x < 40]
        plt.hist(lengths, bins=40)
        plt.xlabel('#Sentence')
        plt.xlabel('Length')
        # plt.plot(x, by2, color='b')
        # plt.plot(x, fy2, color='r')
        # plt.legend(['base', 'full'])

        # plt.subplot(222)
        # x = self.bin([i for i in range(1, 40)], function='max')
        # plt.title('Acc vs #pruned_node ')
        # plt.xlabel('#pruned_node')
        # plt.plot(x, by3, color='b')
        # plt.plot(x, fy3, color='r')
        # plt.legend(['base', 'full'])

        plt.subplot(312)
        x = [i for i in range(41)]
        y = [f - b for b, f in zip(by3, fy3)]

        plt.title('F1 improvement vs #prune-node')
        plt.xlabel('#pruned node')
        plt.ylabel('F-score improvement (%)')
        plt.xlim(start - 1, end)
        plt.bar(x, y)
        y = pd.DataFrame(y).rolling(5, center=True, min_periods=1).mean().to_numpy()
        plt.plot(x, y, color='b', linestyle='dashed')
        plt.legend(['Moving average (k=5)','Improvement by #pruned-node'])
        plt.axhline(y=0, color='black', linestyle='solid')


        plt.subplot(313)

        width = 0.15
        x = np.array([i for i in range(5)])
        y = [f - b for b, f in zip(by4[:5], fy4[:5])]
        # y = [min(i, 5) for i in y]

        plt.title('F1 improvement vs pruning percentage')
        plt.bar(x, y, width=width)
        plt.xticks(x, [str(i * 20) for i in x])
        plt.xlabel('Pruning percentage')
        plt.ylabel('F-score improvement (%)')

        start, end = 1, 20
        lengths, x, by1, by2, by3, by4 = self.extract_length_by_bin(self.base_pred, start, end)
        lengths, x, fy1, fy2, fy3, fy4 = self.extract_length_by_bin(self.full_pred, start, end)
        x = np.array([i for i in range(5)])
        y = [f - b for b, f in zip(by4[:5], fy4[:5])]
        # y = [min(i, 5) for i in y]
        plt.bar(x + 0.2, y, width=width, color='r')

        # start, end = 10, 20
        # lengths, x, by1, by2, by3, by4 = self.extract_length_by_bin(self.base_pred, start, end)
        # lengths, x, fy1, fy2, fy3, fy4 = self.extract_length_by_bin(self.full_pred, start, end)
        # x = np.array([i for i in range(5)])
        # y = [f - b for b, f in zip(by4[:5], fy4[:5])]
        # y = [min(i, 5) for i in y]
        # plt.bar(x + 0.3, y, width=width, color='g')

        start, end = 20, 40
        lengths, x, by1, by2, by3, by4 = self.extract_length_by_bin(self.base_pred, start, end)
        lengths, x, fy1, fy2, fy3, fy4 = self.extract_length_by_bin(self.full_pred, start, end)
        x = np.array([i for i in range(5)])
        y = [f - b for b, f in zip(by4[:5], fy4[:5])]
        # y = [min(i, 5) for i in y]
        plt.bar(x + 0.4, y, width=width, color='orange')

        plt.legend(['All', 'Short','Long'])

        # start, end = 30, 40
        # lengths, x, by1, by2, by3, by4 = self.extract_length_by_bin(self.base_pred, start, end)
        # lengths, x, fy1, fy2, fy3, fy4 = self.extract_length_by_bin(self.full_pred, start, end)
        # x = np.array([i for i in range(5)])
        # y = [f - b for b, f in zip(by4[:5], fy4[:5])]
        # y = [min(i, 5) for i in y]
        # plt.bar(x + 0.60, y, width=width, color='gray')
        plt.axhline(y=0, color='black', linestyle='solid')
        # plt.suptitle('All sentences')

        plt.tight_layout(pad=3)
        # plt.show()
        plt.savefig('images/analysis.png')

if __name__ == '__main__':
    # load_files('base')
    # load_files('full400')
    # load_files('full1000')
    # load_files('full4000')

    base = load_json('checkpoints/base.test.1600.json')
    full = load_json('checkpoints/full4000.test.1600.json')

    a = Analyze(base, full)
    a.analyze()
