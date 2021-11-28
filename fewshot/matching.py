import torch


class MatchingNetwork(torch.nn.Module):
    def __init__(self, encoder, args):
        super(MatchingNetwork, self).__init__()
        self.encoder = encoder
        self.args = args
        self.cosine = torch.nn.CosineSimilarity(dim=-1)

    def init_weight(self):
        pass

    def forward(self, batch, setting):
        B, N, K, Q = setting

        encoded = self.encoder(batch)
        embeddings = encoded['embedding']
        D = embeddings.shape[-1]
        embeddings = embeddings.view(B, N, K + Q, D)
        # print('| Matching > embedding', tuple(embeddings.shape))
        support = embeddings[:, :, :K, :].contiguous()  # B x N x K x D
        query = embeddings[:, :, K:, :].contiguous()  # B x N x Q x D

        prototypes = support.mean(dim=2)  # -> B x N x D
        prototypes = prototypes.unsqueeze(dim=1)  # B x 1 x N x D
        # print('| Matching > prototypes', tuple(prototypes.shape))

        query = query.view(B, -1, D)  # -> B x NQ x D
        query = query.unsqueeze(dim=2)  # -> B x NQ x 1 x D
        # print('| Matching > query', tuple(query.shape))

        similarity = self.cosine(query, prototypes)  # B x NQ x N
        # print('| Matching > similarity', tuple(similarity.shape))

        return_item = {
            'logit': similarity
        }

        for k, v in encoded.items():
            if k != 'embedding':
                return_item[k] = v
        return return_item
