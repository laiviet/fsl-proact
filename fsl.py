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
from fewshot.trainer import FSLTrainer


def main(args):
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = 'cuda'
    args.max_length = dataset_constant[args.dataset]['max_length']
    args.n_class = dataset_constant[args.dataset]['n_class']
    args.train_way = dataset_constant[args.dataset]['train_way']
    args.other = dataset_constant[args.dataset]['other']

    B = args.batch_size  # default = 4
    TN = args.train_way  # default = 20
    N = args.way
    K = args.shot  # default = 5
    Q = args.query  # default = 5
    O = args.other

    current_time = str(datetime.datetime.now().time())
    args.log_dir = 'logs/{}-{}-way-{}-shot-{}'.format(args.model, args.way, args.shot, current_time)

    print('Before load')

    feature_set = feature_map[args.encoder]
    train_dl = FSLDataset(TN, K, Q, O,
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

    fsl_model = fsl_class[args.model](encoder, args)
    # print(fsl_model)
    # exit(0)
    fsl_model.init_weight()
    fsl_model.cuda()

    wsd_model = None
    wsd_train_dl = None
    if args.beta > 0.0:
        wsd_encoder = BertLinear(args=args)
        wsd_feature = feature_map['bertlinear'] + feature_map['gcn'] + ['label_mask']
        state_dict = torch.load('checkpoints/wsd_bertlinear_63.16450397655923.torch')
        wsd_model = WSDModel(wsd_encoder, dataset_constant['semcor']['n_class'], args)
        wsd_model.load_my_state_dict(state_dict)
        wsd_model.freeze_encoder()
        wsd_model.cuda()
        wsd_train_dl = SemcorDataset(features=wsd_feature, prefix='datasets/semcor/supervised/train')
        wsd_train_dl = DataLoader(wsd_train_dl, batch_size=128, shuffle=True, collate_fn=SemcorDataset.pack)

    # if len(device_ids) > 1:
    #     fsl_model = torch.nn.DataParallel(fsl_model, device_ids=device_ids)
    #     if wsd_model is not None:
    #         wsd_model = torch.nn.DataParallel(wsd_model, device_ids=device_ids)

    fsl_trainer = FSLTrainer(fsl_model, train_dl, dev_dl, test_dl, args, wsd_model, wsd_train_dl)
    fsl_trainer.do_train()


if __name__ == '__main__':
    args = argument_parser().parse_args()
    main(args)
