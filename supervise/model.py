import torch
import torch.nn as nn


class ClassificationModel(nn.Module):

    def __init__(self, encoder, n_class, args=None):
        super(ClassificationModel, self).__init__()
        self.encoder = encoder

        input_dim = self.encoder.hidden_size
        self.fc = nn.Sequential(
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Dropout(),
            nn.Linear(input_dim, n_class)
        )

    def forward(self, inputs):
        encoded = self.encoder(inputs)
        embeddings = encoded['embedding']
        logits = self.fc(embeddings)

        return_item = {
            'logit': logits
        }

        for k, v in encoded.items():
            if k != 'embedding':
                return_item[k] = v
        return return_item


class WSDModel(ClassificationModel):

    def __init__(self, encoder, n_class, args):
        super(WSDModel, self).__init__(encoder, n_class, args)

    def freeze_encoder(self):
        for params in self.encoder.parameters():
            params.requires_grad = False

    def forward(self, inputs):
        label_mask = inputs['label_mask']
        x = self.encoder(inputs)
        logits = self.fc(x['embedding'])

        m = torch.abs(logits.detach().min())
        logits = logits + m + 10

        logits = logits * label_mask
        return {
            'logit': logits
        }

    def forward_with_external_encoder(self, inputs, encoder):
        label_mask = inputs['label_mask']
        x = encoder(inputs)
        logits = self.fc(x['embedding'])

        m = torch.abs(logits.detach().min())
        logits = logits + m + 10

        logits = logits * label_mask
        return {
            'logit': logits
        }

    def load_my_state_dict(self, pretrained_dict):
        model_dict = self.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # all_keys = set(model_dict.keys()).union(set(pretrained_dict.keys()))
        # for k in all_keys:
        #     print(k in model_dict, k in pretrained_dict, k)

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.load_state_dict(pretrained_dict)
