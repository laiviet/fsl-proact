import torch.nn as nn
import torch
from sentence_encoder.basenet import BaseNet
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class LSTM(BaseNet):

    def __init__(self, vectors, args):
        super(LSTM, self).__init__(vectors, args)
        self.device = args.device
        self.hidden_size = args.hidden_size
        self.window = args.window
        self.max_length = args.max_length
        self.lstm = nn.LSTM(input_size=self.embedder.hidden_size,
                          hidden_size=args.hidden_size,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)
        self.linear = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.non_linear = nn.Tanh()


    def init_weight(self):
        self.apply(init_weights)

    def initHidden(self, batch_size):
        h0 = torch.zeros(2, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(2, batch_size, self.hidden_size).to(self.device)
        return h0,c0

    def select_anchor(self, emb, anchor_index):
        """

        :param emb: B x L x D
        :param anchor_index: B
        :return:
        """
        B, L, D = emb.shape
        u = torch.tensor([x for x in range(L)]).unsqueeze(0).to(self.device)
        v = anchor_index.view(B, 1)
        mask = (u == v).unsqueeze(dim=2).to(self.device)
        x = torch.masked_select(emb, mask).view(-1, D)
        return x

    def forward(self, inputs):
        embedding = self.embedder(inputs)
        shape = list(embedding.shape)

        embedding = embedding.view(tuple([-1] + shape[-2:]))
        anchors = inputs['anchor_index'].view(-1)

        # print('| LSTMEncoder: anchors > ', tuple(anchors.shape))
        # print('| LSTMEncoder: embedding > ', tuple(embedding.shape))
        b = 1
        for x in shape[:-2]:
            b *= x
        h0 = self.initHidden(b)
        hidden_states, _ = self.lstm(embedding, h0)

        # print('| LSTMEncoder: hidden_states > ', tuple(hidden_states.shape))

        rnnRep = self.select_anchor(hidden_states, anchors)
        # print('| LSTMEncoder: rnnRep > ', tuple(rnnRep.shape))

        x = rnnRep.view(tuple(shape[:-2] + [2 * self.hidden_size]))
        # print('| LSTMEncoder: x > ', tuple(x.shape))
        x = self.linear(x)
        x = self.non_linear(x)

        return {'embedding': x}
