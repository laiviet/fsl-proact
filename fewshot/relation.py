import torch


class RelationNetwork(torch.nn.Module):
    def __init__(self, encoder, args):
        super(RelationNetwork, self).__init__()
        self.encoder = encoder
        self.scoring = torch.nn.Linear(4 * args.hidden_size, 1)

    def init_weight(self):
        torch.nn.init.xavier_uniform_(self.scoring.weight)
        self.scoring.bias.data.fill_(0.01)

    def forward(self, batch, setting):
        B, N, K, Q = setting

        encoded = self.encoder(batch)
        embeddings = encoded['embedding']
        D = embeddings.shape[-1]
        embeddings = embeddings.view(B, N, K + Q, D)
        # print('| Proto > embedding', tuple(embeddings.shape))
        support = embeddings[:, :, :K, :].contiguous()  # B x N x K x D
        query = embeddings[:, :, K:, :].contiguous()  # B x N x Q x D

        prototypes = support.mean(dim=2)  # -> B x N x D
        prototypes = prototypes.unsqueeze(dim=1)  # B x 1 x N x D
        prototypes = prototypes.expand(-1, N * Q, -1, -1)  # B x NQ x N x D

        query = query.view(B, -1, D)  # -> B x NQ x D
        query = query.unsqueeze(dim=2)  # -> B x NQ x 1 x D
        query = query.expand(-1, -1, N, -1)  # B x NQ x N x D

        abs = torch.abs(prototypes - query)
        cosine = prototypes * query

        sim = torch.cat([prototypes, query, abs, cosine], dim=3)  # B x NQ x N x 4D
        logits = self.scoring(sim).squeeze(dim=3)  # B x NQ x N
        return_item = {
            'logit': logits
        }

        for k, v in encoded.items():
            if k != 'embedding':
                return_item[k] = v
        return return_item
