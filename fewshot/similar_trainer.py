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


class SimilarFSLTrainer(torch.nn.Module):

    def __init__(self, fsl_model, loss_fn, train_dl, dev_dl, test_dl, args, wsd_model=None, wsd_dl=None):
        super(SimilarFSLTrainer, self).__init__()
        self.device = args.device
        self.fsl_model = fsl_model
        self.wsd_model = wsd_model
        self.wsd_dl = wsd_dl
        self.TN = args.train_way
        self.N = args.way
        self.K = args.shot
        self.Q = args.query
        self.B = args.batch_size
        self.train_dl = train_dl
        self.dev_dl = dev_dl
        self.test_dl = test_dl
        self.ignore_cuda_feature = ['token', 'label', 'target']

        self.args = args
        self.loss_fn = loss_fn

        fsl_params = [x for x in self.fsl_model.parameters() if x.requires_grad]
        wsd_params = [x for x in self.wsd_model.parameters() if x.requires_grad]

        self.fsl_optimizer = torch.optim.SGD(fsl_params, lr=self.args.lr)
        self.wsd_optimizer = torch.optim.Adam(wsd_params, lr=0.0005)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.fsl_optimizer, gamma=0.1, step_size=self.args.lr_step_size)

    def do_train(self):
        self.fsl_model.train()
        self.wsd_model.train()

        self.train_iter = iter(self.train_dl)
        self.wsd_iter = iter(self.wsd_dl)

        i=0
        while True:
            if random.random()< 0.2:
                self.train_fsl(i)
                i+=1
            else:
                self.train_wsd(i)

            # Evaluation
            train_iter = 10 if self.args.debug else 400
            if i % train_iter == 0 and i > 0:
                self.do_eval('dev', i)
                self.do_eval('test', i)
                self.fsl_model.train()
            if i > 6000:
                return

    def train_fsl(self, i):
        # print('train FSL')
        B, N, K, Q = self.B, self.TN + 1, self.K, self.Q
        setting = (self.B, self.TN + 1, self.K, self.Q)
        batch = next(self.train_iter)
        for k, v in batch.items():
            if k not in self.ignore_cuda_feature:
                batch[k] = v.to(self.device)
                # print(k, v.shape)

        return_item = self.fsl_model(batch, setting)
        target = batch['target'].view(-1).cuda()
        logits = return_item['logit'].view(-1, N)
        total_loss = 0.0
        loss = self.loss_fn(logits, target)

        mutual_loss = self.args.alpha * return_item['mutual_loss']
        similar_loss = self.wsd_mutual(batch)
        total_loss = loss + self.args.alpha * mutual_loss+ self.args.beta * similar_loss

        total_loss.backward()
        self.fsl_optimizer.step()
        self.scheduler.step()
        train_log_iter = 5 if self.args.debug else 400
        if i % train_log_iter == 0 and i > 0:
            output_format = '| @ {} Loss={:.4f} Tree={:.4f} SIM={:.4f}'
            print(output_format.format(i, loss.item(),
                                       mutual_loss.item(),
                                       similar_loss.item()))


    def train_wsd(self, i):
        # print('| Train wsd')
        batch = next(self.wsd_iter,None)
        if batch is None:
            self.wsd_iter=iter(self.wsd_dl)
            batch = next(self.wsd_iter)

        self.wsd_optimizer.zero_grad()
        for k, v in batch.items():
            if k not in self.ignore_cuda_feature:
                batch[k] = batch[k].to(self.device)
        return_item = self.wsd_model(batch)
        target = batch['target'].to(self.device)
        logits = return_item['logit']

        wsd_loss = self.loss_fn(logits, target)
        similar_loss = self.wsd_mutual(batch)

        total_loss = wsd_loss+ self.args.beta * similar_loss
        train_log_iter = 5 if self.args.debug else 400
        if i % train_log_iter == 0 and i > 0:
            print('WSD={} SIM={}'.format(wsd_loss.item(), similar_loss.item()))

        total_loss.backward()
        self.wsd_optimizer.step()

    def wsd_mutual(self, batch):
        p_gcn = self.fsl_model.encoder(batch)
        p_wsd = self.wsd_model.encoder(batch)

        p = torch.softmax(p_gcn['embedding'], dim=-1)
        q = torch.softmax(p_wsd['embedding'], dim=-1)

        # mutual_loss = torch.sum(p * ((p / q + 1e-10).log()))
        mutual_loss = torch.sum((p - q) ** 2)
        return mutual_loss

    def get_detach_loss(self, loss):
        if loss:
            return loss.detach().cpu().numpy()
        else:
            return 0.0

    def do_eval(self, part='test', iteration=0):
        if part == 'dev':
            dl = self.dev_dl
        else:
            dl = self.test_dl
        B, N, K, Q = self.B, self.N + 1, self.K, self.Q
        setting = (self.B, self.N + 1, self.K, self.Q)
        self.fsl_model.eval()

        with torch.no_grad():
            predictions = []
            targets = []
            sample_indices = []
            for i, batch in enumerate(dl):
                for k, v in batch.items():
                    if k not in self.ignore_cuda_feature:
                        batch[k] = batch[k].cuda()
                return_item = self.fsl_model(batch, setting)
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

    def metrics(self, targets, predictions):

        def avg(x):
            return sum(x) / len(x)

        precisions, recalls, fscores = [], [], []

        labels = [x for x in range(1, self.N + 1)]

        # print(len(targets))

        for _t, _p in zip(targets, predictions):
            p = 100 * precision_score(_t, _p, labels=labels, average='micro')
            r = 100 * recall_score(_t, _p, labels=labels, average='micro')
            f = 100 * f1_score(_t, _p, labels=labels, average='micro')
            precisions.append(p)
            recalls.append(r)
            fscores.append(f)
        return avg(precisions), avg(recalls), avg(fscores)

    def fscore(self, targets, predictions):
        targets = numpy.concatenate(targets)
        predictions = numpy.concatenate(predictions)
        zeros = np.zeros(predictions.shape, dtype='int')
        numPred = np.sum(np.not_equal(predictions, zeros))
        numKey = np.sum(np.not_equal(targets, zeros))
        predictedIds = np.nonzero(predictions)
        preds_eval = predictions[predictedIds]
        keys_eval = targets[predictedIds]
        correct = np.sum(np.equal(preds_eval, keys_eval))
        # print('correct : {}, numPred : {}, numKey : {}'.format(correct, numPred, numKey))
        precision = 100.0 * correct / numPred if numPred > 0 else 0.0
        recall = 100.0 * correct / numKey
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0. else 0.0
        return precision, recall, f1
