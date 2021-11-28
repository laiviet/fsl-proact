import argparse

from custom_dataset import *

from constant import *
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from fewshot import *


def str_list(text):
    return tuple(text.split(','))


def int_list(text):
    return [int(x) for x in text.split(',')]


def parse_int_list(input_):
    if input_ == None:
        return []
    return list(map(int, input_.split(',')))


def parse_float_list(input_):
    if input_ == None:
        return []
    return list(map(float, input_.split(',')))


def one_or_list(parser):
    def parse_one_or_list(input_):
        output = parser(input_)
        if len(output) == 1:
            return output[0]
        else:
            return output

    return parse_one_or_list


def argument_parser():
    parser = argparse.ArgumentParser()
    # Training setting
    parser.add_argument('-m', '--model', default='svmww', choices=meta_opt_net_class.keys())
    parser.add_argument('-e', '--encoder', default='gcn', choices=encoder_class.keys())
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('-o', '--optimizer', default='sgd', type=str, choices=['adam', 'sgd', 'adadelta'])
    parser.add_argument('--lr', default=0.2, type=float)
    parser.add_argument('--lr_step_size', default=1000, type=int)

    # Few-shot settings
    parser.add_argument('-d', '--dataset', default='ace', choices=['ace', 'rams', 'fed', 'debug'])
    parser.add_argument('--save', default='checkpoints', type=str)
    parser.add_argument('--seed', default=1234, type=int)

    parser.add_argument('-t', '--train_way', default=20, type=int)
    parser.add_argument('-n', '--way', default=5, type=int)
    parser.add_argument('-k', '--shot', default=5, type=int)
    parser.add_argument('-q', '--query', default=4, type=int)

    # Embedding
    parser.add_argument('--bert_layer', default=12, type=int)
    parser.add_argument('--embedding', default='glove', type=str_list)
    parser.add_argument('--tune_embedding', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')

    return parser


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_data(args):
    TN, N, K, Q = args.train_way, args.way, args.shot, args.query
    bert_layer = args.bert_layer
    test_dl = EmbeddingFSLDataset(N, K, Q, 500,
                                  'datasets/{}/fsl/test'.format(args.dataset),
                                  bert_layer=bert_layer)

    fsl_pack = EmbeddingFSLDataset.fsl_pack
    test_emb = torch.nn.Embedding(test_dl.embedding.shape[0], test_dl.embedding.shape[1],
                                  _weight=test_dl.embedding).to(device)
    test_emb.weight.requires_grad = False
    test_dl = DataLoader(test_dl, batch_size=3, num_workers=2, shuffle=False, collate_fn=fsl_pack)

    if args.debug:
        return test_dl, test_dl, test_dl, test_emb, test_emb, test_emb

    train_dl = EmbeddingFSLDataset(N, K, Q, 400,
                                   'datasets/{}/fsl/train'.format(args.dataset),
                                   bert_layer=bert_layer)
    train_emb = torch.nn.Embedding(train_dl.embedding.shape[0], train_dl.embedding.shape[1],
                                   _weight=train_dl.embedding).to(device)
    train_emb.weight.requires_grad = False
    train_dl = DataLoader(train_dl, batch_size=3, num_workers=2, collate_fn=fsl_pack)

    dev_dl = EmbeddingFSLDataset(N, K, Q, 500,
                                 'datasets/{}/fsl/dev'.format(args.dataset),
                                 bert_layer=bert_layer)
    dev_emb = torch.nn.Embedding(dev_dl.embedding.shape[0], dev_dl.embedding.shape[1],
                                 _weight=dev_dl.embedding).to(device)
    dev_emb.weight.requires_grad = False
    dev_dl = DataLoader(dev_dl, batch_size=3, num_workers=2, shuffle=False, collate_fn=fsl_pack)

    return train_dl, dev_dl, test_dl, train_emb, dev_emb, test_emb


def main(args):
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda'
    print('Before load')

    train_dl, dev_dl, test_dl, train_emb, dev_emb, test_emb = load_data(args)
    model = meta_opt_net_class[args.model](device=device)

    print('Head class', meta_opt_net_class[args.model])

    for epoch in range(20):
        # pbar = tqdm.tqdm(train_dl)
        # for batch in pbar:
        #     sx, sy = batch['support_set'].to(device), batch['support_targets'].to(device)
        #     qx, qy = batch['query_set'].to(device), batch['query_targets'].to(device)
        #
        #     # x = torch.cat([sx, qx], -1)
        #     # y = torch.cat([sy, qy], -1)
        #     #
        #     sx_emb = train_emb(sx)
        #     qx_emb = train_emb(qx)
        #
        #     p, r, f1 = model.fit_and_evaluate(sx_emb, sy, qx_emb, qy)
        #     loss = 0.0
        #
        #     pbar.set_description(f'F1={f1:.4f}')
        #
        dev_res, test_res = [], []
        for idev, batch in tqdm.tqdm(enumerate(dev_dl)):
            sx, sy = batch['support_set'].to(device), batch['support_targets'].to(device)
            qx, qy = batch['query_set'].to(device), batch['query_targets'].to(device)

            sx_emb = dev_emb(sx)
            qx_emb = dev_emb(qx)
            perf, loss = model.fit_and_evaluate(sx_emb, sy, qx_emb, qy)
            dev_res.append(perf)
        for idev, batch in tqdm.tqdm(enumerate(test_dl)):
            sx, sy = batch['support_set'].to(device), batch['support_targets'].to(device)
            qx, qy = batch['query_set'].to(device), batch['query_targets'].to(device)
            sx_emb = test_emb(sx)
            qx_emb = test_emb(qx)
            perf, loss = model.fit_and_evaluate(sx_emb, sy, qx_emb, qy)
            test_res.append(perf)

        dev = np.mean(np.array(dev_res), axis=0) * 100
        test = np.mean(np.array(test_res), axis=0) * 100
        print(f'{dev[0]:.2f} {dev[1]:.2f} {dev[2]:.2f} {test[0]:.2f} {test[1]:.2f} {test[2]:.2f}')


if __name__ == '__main__':
    args = argument_parser().parse_args()
    main(args)
