import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support

class LinearTransform(nn.Module):

    def __init__(self, input_size, device='cuda'):
        super(LinearTransform, self).__init__()
        self.device = device
        self.input_size = input_size
        self.model = nn.Sequential(
            nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(input_size, 1024),
            nn.Tanh()
        )
        nn.init.xavier_uniform_(self.model[2].weight)

    def forward(self, emb):
        return self.model(emb)


class TransformerTransform(nn.Module):

    def __init__(self, input_size):
        super(TransformerTransform, self).__init__()
        self.input_size = input_size
        self.nhead = 8
        self.d_model = 512
        print('D_model: ', self.d_model)
        print('nhead: ', self.nhead)
        layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.linear = LinearTransform(input_size)

    def forward(self, emb):
        # print('Embedding shape: ', emb.shape)
        C, K, D = emb.shape

        assert D == self.input_size
        x = emb.view(C * K, -1, self.d_model)  # CK x head x dmodel

        # print('TransformerTransform > x view: ', x.shape)
        x = self.encoder(x)
        # print('TransformerTransform > x encoder: ', x.shape)
        x = x.view(C, K, -1)  # C x K x 1024
        # print('TransformerTransform > x final: ', x.shape)

        x = self.linear(x)
        return x


class LogisticRegression(torch.nn.Module):

    def __init__(self, input_size=3072, device='cuda'):
        super(LogisticRegression, self).__init__()
        self.device = device
        self.transfomer = LinearTransform(input_size=input_size)
        # self.transfomer = TransformerTransform(input_size=input_size)
        self.ce = torch.nn.CrossEntropyLoss()

    def proto_loss(self, x):
        """

        :param x: C x KQ x D
        :return:
        """
        C, K, D = x.shape

        # print(x.shape)
        support = x[:, :-2, :]
        prototypes = support.mean(dim=1).view(1, C, D)  # 1 x C x D
        query = x[:, -2:, :].contiguous().view(-1, 1, D)  # 2C x 1 x D

        distance = query - prototypes  # 2C x C x D
        square = distance * distance
        sum_square = square.sum(-1)  # 2C x C

        targets = torch.LongTensor([[x, x] for x in range(C)]).view(-1).to(self.device)

        loss = self.ce(-sum_square, targets)
        return loss

    def fit(self, emb, y, max_epoch=1000, eval=False):
        C = emb.shape[0]
        y = y.view(-1)
        self.fc = nn.Linear(1024, C).to(self.device)
        torch.nn.init.xavier_uniform_(self.fc.weight)

        if eval:
            optimizer = torch.optim.SGD(self.fc.parameters(), lr=0.1)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

        for e in range(max_epoch):
            # for epoch in range(max_epoch):
            optimizer.zero_grad()
            x = self.transfomer(emb)  #
            logits = self.fc(x)
            # print("| fit > logits", tuple(logits.shape))
            logits = logits.view(-1, C)
            # print("| fit > logits", tuple(logits.shape))
            predictions = torch.argmax(logits, dim=1)
            # print("| fit > predictions", tuple(predictions.shape))
            loss = self.ce(logits, y)
            # proto_loss = 0.1 * self.proto_loss(h)
            # (loss + proto_loss).backward()
            loss.backward()
            optimizer.step()
            if torch.sum(predictions - y) == 0:
                return loss.detach().cpu().numpy(), 0
        return loss.detach().cpu().numpy(), 0

    def evaluate(self, x, y):
        C = x.shape[0]
        x = self.transfomer(x)
        #
        logits = self.fc(x)
        # logits = self.fc(x)
        logits = logits.view(-1, C)
        predictions = torch.argmax(logits, dim=1)

        labels = [x for x in range(1, C)]
        y = y.view(-1).cpu().numpy()
        predictions = predictions.view(-1).cpu().numpy()

        p, r, f, s = precision_recall_fscore_support(y, predictions, labels=labels, average='micro')

        return f
