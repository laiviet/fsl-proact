import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import random
import datetime

from custom_dataset import *
from argument_parser import *


def main(args):
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = 'cuda'
    # torch.cuda.set_device(0)

    args.max_length = dataset_constant[args.dataset]['max_length']
    args.n_class = dataset_constant[args.dataset]['n_class']
    args.train_way = dataset_constant[args.dataset]['train_way']
    args.other = dataset_constant[args.dataset]['other']

    B = args.batch_size  # default = 4
    TN = dataset_constant[args.dataset]['train_way']
    N = args.way
    K = args.shot  # default = 5
    Q = args.query  # default = 5
    O = dataset_constant[args.dataset]['other']

    current_time = str(datetime.datetime.now().time())
    args.log_dir = 'logs/{}-{}-way-{}-shot-{}'.format(args.model, args.way, args.shot, current_time)

    print('Before load')

    feature_set = feature_map[args.encoder]
    train_dl = FSLDataset(TN, K * 2, Q * 2, O,
                          features=feature_set,
                          length=1000000,
                          prefix='datasets/{}/fsl/train'.format(args.dataset))
    dev_dl = FSLDataset(N, K, Q, O,
                        features=feature_set,
                        length=500,
                        prefix='datasets/{}/fsl/dev'.format(args.dataset))
    test_dl = FSLDataset(N, K, Q, O,
                         features=feature_set,
                         length=500,
                         prefix='datasets/{}/fsl/test'.format(args.dataset))

    train_dl = DataLoader(train_dl, batch_size=B, num_workers=8, collate_fn=FSLDataset.fsl_pack)
    dev_dl = DataLoader(dev_dl, batch_size=B, num_workers=8, collate_fn=FSLDataset.fsl_pack, shuffle=False)
    test_dl = DataLoader(test_dl, batch_size=B, num_workers=8, collate_fn=FSLDataset.fsl_pack, shuffle=False)

    print('-' * 80)
    for k, v in args.__dict__.items():
        print('{}\t{}'.format(k, v))
    print('-' * 80)

    _, vectors = load_pickle('datasets/vocab.pkl')
    encoder = encoder_class[args.encoder](vectors=vectors, args=args)
    encoder.init_weight()

    if args.model == 'melr':
        fsl_model = MELRNetwork(encoder, args)
    elif args.model == 'melrplus':
        fsl_model = MELRPlus(encoder, args)
    else:
        print('Not a MELR based model')

    fsl_model.init_weight()
    fsl_model.cuda()

    fsl_trainer = MELRFSLTrainer(fsl_model, train_dl, dev_dl, test_dl, args)
    fsl_trainer.do_train()


if __name__ == '__main__':
    args = argument_parser().parse_args()
    main(args)
