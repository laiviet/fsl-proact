import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import OrderedDict
from sentence_encoder.embedding import Embedding
from sentence_encoder.utils import init_weights

def retrieveLocalEmbeddings(embed, anchors, window, device):
    zeros = torch.zeros(embed.size(0), window, embed.size(2)).float().cuda()

    padded = torch.cat([zeros, embed, zeros], 1)

    ids = []
    for i in range(2 * window + 1):
        ids += [(anchors + i).long().view(-1, 1)]
    ids = torch.cat(ids, 1)
    ids = ids.unsqueeze(2).expand(ids.size(0), ids.size(1), embed.size(2)).cuda()

    res = padded.gather(1, ids)
    res = res.view(res.size(0), -1)
    return res


def clipTwoDimentions(mat, norm=3.0, device='cpu'):
    col_norms = ((mat ** 2).sum(0, keepdim=True)) ** 0.5
    desired_norms = col_norms.clamp(0.0, norm)
    scale = desired_norms / (1e-7 + col_norms)
    res = mat * scale
    res = res.cuda()
    return res


class BaseNet(nn.Module):

    def __init__(self, vectors, args):
        super(BaseNet, self).__init__()
        self.args = args
        self.device = args.device
        self.hidden_size = args.hidden_size
        self.cnn_kernel_sizes = [2, 3, 4, 5]
        self.cnn_kernel_number = 150

        self.embedder = Embedding(vectors,
                                  max_length=args.max_length,
                                  tune_embedding=args.tune_embedding,
                                  device=args.device)
        self.window = args.window
        self.embedding_input_dim = self.embedder.hidden_size
        self.conv = nn.Conv1d(self.embedding_input_dim, self.hidden_size, 3, padding=1)
        self.pool = nn.MaxPool1d(args.max_length)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(args.dropout)

    def init_weight(self):
        self.fc.apply(init_weights)

    def createFcModule(self, dim_rep):

        if self.window > 0:
            dim_rep += (1 + 2 * self.window) * self.embedder.hidden_size

        rep_hids = [dim_rep, self.hidden_size, self.args.n_class]

        ofcs = OrderedDict()
        for i, (ri, ro) in enumerate(zip(rep_hids, rep_hids[1:])):
            ofcs['finalRep_' + str(i)] = nn.Linear(ri, ro)
            # ofcs['finalNL_' + str(i)] = nn.Tanh()
        self.fc = nn.Sequential(ofcs)

    def introduceLocalEmbedding(self, frep, inputs):

        assert type(frep) == list

        inWord_embeddings = self.embedder(inputs['indices'], inputs['dist'], inputs['mask'])
        shape = inWord_embeddings.shape

        # print('| Basenet > introduceLocalEmbedding > inWord_embeddings', tuple(inWord_embeddings.shape))

        if self.window > 0:
            local_rep = retrieveLocalEmbeddings(inWord_embeddings.view(-1, shape[-2], shape[-1]),
                                                inputs['anchor_index'], self.window,
                                                self.device)
            frep += [local_rep]

        frep = frep[0] if len(frep) == 1 else torch.cat(frep, 1)

        return frep


#### Pooling Methods ######

def pool_anchor(rep, iniPos, mask, nmask, device):
    ids = iniPos.view(-1, 1)
    ids = ids.expand(ids.size(0), rep.size(2)).unsqueeze(1).cuda()

    res = rep.gather(1, ids)
    res = res.squeeze(1)
    return res


def pool_max(rep, iniPos, mask, nmask, device):
    rep = torch.exp(rep) * mask.unsqueeze(2)
    res = torch.log(rep.max(1)[0])
    return res


def pool_dynamic(rep, iniPos, mask, nmask, device):
    rep = torch.exp(rep) * mask.unsqueeze(2)
    left, right = [], []
    batch, lent, dim = rep.size(0), rep.size(1), rep.size(2)
    for i in range(batch):
        r, id, ma = rep[i], iniPos.tolist()[i], mask[i]
        tleft = torch.log(r[0:(id + 1)].max(0)[0].unsqueeze(0))
        left += [tleft]
        if (id + 1) < lent and ma.cpu().numpy()[(id + 1):].sum() > 0:
            tright = torch.log(r[(id + 1):].max(0)[0].unsqueeze(0))
        else:
            tright = Variable(torch.zeros(1, dim).cuda())
        right += [tright]
    left = torch.cat(left, 0)
    right = torch.cat(right, 0)
    res = torch.cat([left, right], 1)

    return res


def pool_entity(rep, iniPos, mask, nmask, device):
    rep = torch.exp(rep) * nmask.unsqueeze(2)
    res = torch.log(rep.max(1)[0])
    return res


### KL Loss

def kl(logits1, logits2):
    p1 = F.softmax(logits1, dim=-1)
    p2 = F.softmax(logits2, dim=-1)
    loss = torch.sum(p1 * ((p1 / p2).log()))
    return loss
