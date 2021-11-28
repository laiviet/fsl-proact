import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AMSLossv1(nn.Module):

    def __init__(self, m=0.4):
        '''
        AM Softmax Loss
        '''
        super(AMSLossv1, self).__init__()
        self.m = m

    def forward(self, logits, targets):
        '''
        input shape (N, in_features)
        '''
        numerator = torch.diagonal(logits.transpose(0, 1)[targets])
        excl = torch.cat([torch.cat((logits[i, :y], logits[i, y + 1:])).unsqueeze(0) for i, y in enumerate(targets)],
                         dim=0)
        excl = excl + self.m
        denominator = torch.exp(numerator) + torch.sum(torch.exp(excl), dim=1)

        print(numerator.sum())
        print(denominator.sum())
        L = numerator - torch.log(denominator + 1e-10)
        return -torch.mean(L)


class AMSLoss(nn.Module):

    def __init__(self, m=1.0):
        super(AMSLoss, self).__init__()
        self.m = m
        self.one_minus_exp_m = 1.0 - math.exp(m)

    def forward(self, logits, labels, eps=1e-10):
        """

        :param logits: B x C
        :param targets: B
        :return:
        """
        numerator = torch.diagonal(logits.transpose(0, 1)[labels])
        denominator = torch.sum(torch.exp(logits + self.m), dim=1) + torch.exp(numerator) * self.one_minus_exp_m
        L = numerator - torch.log(denominator + eps)
        return -torch.mean(L)


def test():
    import random
    random.seed(1)
    B = 6
    N = 5
    l = AMSLoss(0.2)
    l2 = AMSLossv2(0.2)
    print(l)

    x = torch.FloatTensor([0, 10, 20, 30, 40]).repeat(B, 1)
    x = torch.FloatTensor([1, 1, 1, 1, 1]).repeat(B, 1)
    print(x)
    labels = [random.randint(0, N - 1) for _ in range(B)]
    labels = [4 for _ in range(B)]
    labels = torch.LongTensor(labels)
    print(labels)

    print('Lv1: ', l(x, labels))
    print('Lv2: ', l2(x, labels))

    ce = nn.CrossEntropyLoss()
    print(ce(x, labels))


if __name__ == '__main__':
    test()
