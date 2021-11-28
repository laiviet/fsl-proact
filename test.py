import random
from sklearn.metrics import *

if __name__ == '__main__':
    a = [random.randint(0, 5) for i in range(200)]
    b = [random.randint(0, 5) for i in range(200)]

    print(precision_recall_fscore_support(a, b, average='micro'))
    print(precision_recall_fscore_support(a, b, average='micro', labels=[1, 2, 3, 4]))
    print(precision_recall_fscore_support(a, b, average='micro', labels=[0, 1, 2, 3, 4]))

    a = [x + 1 for x in a]
    b = [x + 1 for x in b]
    print(precision_recall_fscore_support(a, b, average='micro', labels=[1, 2, 3, 4, 5]))
    print(precision_score(a, b, average='micro', labels=[1, 2, 3, 4, 5]))
    print(recall_score(a, b, average='micro', labels=[1, 2, 3, 4, 5]))
    print(f1_score(a, b, average='micro', labels=[1, 2, 3, 4, 5]))
