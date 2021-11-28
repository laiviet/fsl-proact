import torch
import numpy
import numpy as np

import torch.nn as nn
import json
import random
from sklearn.metrics import precision_score, recall_score, f1_score
from fewshot.amsl import AMSLoss
import time
from preprocess.utils import save_json
from fewshot.trainer import FSLTrainer


class MELRFSLTrainer(FSLTrainer):

    def __init__(self, fsl_model, train_dl, dev_dl, test_dl, args):
        super(MELRFSLTrainer, self).__init__(fsl_model, train_dl, dev_dl, test_dl, args)


    def do_train(self):
        B, N, K, Q = self.train_setting
        params = [x for x in self.parameters() if x.requires_grad]

        trainable = 0
        for p in params:
            trainable += p.numel()

        print('Trainable params: ', trainable)
        print(f'Settings: B={B} TN={N} K={K} Q={Q}')

        print('Optimizer: SGD')
        optimizer = torch.optim.SGD(params, lr=self.args.lr)
        # optimizer = torch.optim.Adadelta(params, lr=self.args.lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=self.args.lr_step_size)

        self.fsl_model.train()
        for i, batch in enumerate(self.train_dl):
            for k, v in batch.items():
                if k not in self.ignore_cuda_feature:
                    batch[k] = batch[k].cuda()

            if i == 0:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        print(k, tuple(v.shape))
                    elif isinstance(v, list):
                        print(k, 'list of ', len(v), type(v[0]))
            optimizer.zero_grad()
            return_item = self.fsl_model(batch, self.train_setting, training=True)
            target = batch['target'].view(-1).cuda()
            logits = return_item['logit'].view(-1, N)

            # print('Trainer > logits', tuple(logits.shape))
            # print('Trainer > target', tuple(target.shape))
            #
            # for x,y in zip(logits.detach().cpu().numpy(), target):
            #     print(x,y)
            loss = self.ce(logits, target)
            cerc_loss = return_item['cerc_loss']

            total_loss = loss + self.args.alpha * cerc_loss

            total_loss.backward()
            optimizer.step()

            # if scheduler:
            #     scheduler.step()

            # Evaluation
            if self.args.debug:
                train_log_iter = 2
                train_iter = 2
            else:
                train_log_iter = 400
                train_iter = 400

            if i % train_log_iter == 0 and i > 0:
                l = loss.detach().cpu().numpy()
                c = cerc_loss.detach().cpu().numpy()
                t = total_loss.detach().cpu().numpy()
                print(f'| @ {i} FSL={l:.4f} CERC={c:.4f} Total:={t:4f}')

            if i % train_iter == 0 and i > 0:
                self.do_eval('dev', i)
                self.do_eval('test', i)
                self.fsl_model.train()
            if i > 6000:
                return

    def do_eval(self, part='test', iteration=0):
        if part == 'dev':
            dl = self.dev_dl
        else:
            dl = self.test_dl
        B, N, K, Q = self.eval_setting

        self.fsl_model.eval()

        with torch.no_grad():
            predictions = []
            targets = []
            sample_indices = []
            for i, batch in enumerate(dl):
                for k, v in batch.items():
                    if k not in self.ignore_cuda_feature:
                        batch[k] = batch[k].cuda()
                return_item = self.fsl_model(batch, self.eval_setting, training=False)
                logits = return_item['logit'].view(-1, N)
                _pred = torch.argmax(logits, dim=1).view(-1).cpu().numpy()
                _target = batch['target'].view(-1).numpy()

                predictions.append(_pred)
                targets.append(_target)
                sample_indices.append(batch['i'].tolist())

            p, r, f = self.metrics(targets, predictions)
            print('-> {:4s}: {:6.2f} {:6.2f} {:6.2f}'.format(part, p, r, f))

        # Save for further analysize
        predictions = np.stack(predictions).tolist()
        targets = np.stack(targets).tolist()
        item = {'iteration': iteration,
                'sample': sample_indices,
                'target': targets,
                'prediction': predictions}
        path = 'checkpoints/{}.{}.{}.json'.format(self.args.ex, part, iteration)
        save_json(item, path)
