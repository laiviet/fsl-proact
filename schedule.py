#!/disk/vietl/.miniconda3/envs/py36/bin/python

def append(text, path):
    with open(path, 'a') as f:
        f.write(text + '\n')


PROTO = 'proto'
ATTPROTO = 'attproto'

CNN = 'cnn'
GCN = 'gcn'
BERTGCN = 'bertgcn'
BERTCNN = 'bertcnn'
BERTMLP = 'bertmlp'
BERTED = 'berted'

DF_OPT = 'sgd'
DF_LR = 0.0002
DF_BATCHSIZE = 2
DF_DATASET = 'fed'

ZERO = '0'
ALL = 'all'


def schedule(model=PROTO,
             opt=DF_OPT,
             encoder=BERTGCN,
             layer=4,
             dataset=DF_DATASET,
             N=5,
             K=5,
             seeds='1111',
             alphas=ZERO,
             betas=ZERO,
             gammas=ZERO,
             omegas=ZERO,
             xis=ZERO
             ):
    if alphas == ALL:
        alphas = [0, 200, 400, 800]
    else:
        alphas = [int(x) for x in alphas.split(',')]

    if betas == ALL:
        betas = [0, 0.01, 0.001]
    else:
        betas = [float(x) for x in betas.split(',')]

    if gammas == ALL:
        gammas = [0, 200, 400, 800]
    else:
        gammas = [float(x) for x in gammas.split(',')]

    if omegas == ALL:
        omegas = [0, 200, 400, 800]
    else:
        omegas = [float(x) for x in omegas.split(',')]

    if 'bert' in encoder:
        lr = 0.0002
        batch_size = 2
    else:
        lr = 0.001
        batch_size = 4

    seeds = [int(x) for x in seeds.split(',')]

    xis = [float(x) for x in xis.split(',')]

    for alpha in alphas:
        for beta in betas:
            for xi in xis:
                for seed in seeds:
                    log = 'logs/{dataset}/{encoder}/{N}-{K}.{model}.{encoder}.l-{layer}.{opt}-{lr}.a-{alpha}.b-{beta}.xi-{xi}.s-{seed}.txt'.format(
                        dataset=dataset,
                        encoder=encoder,
                        N=N,
                        K=K,
                        model=model,
                        layer=layer,
                        opt=opt,
                        lr=lr,
                        alpha=alpha,
                        beta=beta,
                        xi=xi,
                        seed=seed
                    )

                    err = '{N}-{K}.{model}.l-{layer}.{opt}-{lr}.a-{alpha}.b-{beta}.xi-{xi}.err'.format(
                        dataset=dataset,
                        encoder=encoder,
                        N=N,
                        K=K,
                        model=model,
                        layer=layer,
                        opt=opt,
                        lr=lr,
                        alpha=alpha,
                        beta=beta,
                        xi=xi
                    )

                    command = 'python fsl.py -n {N} -k {K} -m {model} -d {dataset} -o {opt} --lr {lr} -e {encoder} --seed {seed}'.format(
                        dataset=dataset,
                        encoder=encoder,
                        model=model,
                        N=N,
                        K=K,
                        layer=layer,
                        opt=opt,
                        lr=lr,
                        seed=seed
                    ) + ' --alpha {alpha} --beta {beta} --xi {xi} -b {batch_size} > {log} 2> {err}'.format(
                        alpha=alpha,
                        beta=beta,
                        xi=xi,
                        batch_size=batch_size,
                        log=log,
                        err=err
                    )
                    append(command, 'commands.txt')


def schedule_supervise_learning(opt=DF_OPT,
                                encoder=BERTGCN,
                                layer=4,
                                lr=ZERO,
                                dataset=DF_DATASET,
                                seeds='1111',
                                alphas=ZERO,
                                betas=ZERO
                                ):
    alphas = [int(x) for x in alphas.split(',')]
    betas = [float(x) for x in betas.split(',')]
    learning_rates = [float(x) for x in lr.split(',')]
    seeds = [int(x) for x in seeds.split(',')]

    batch_size = 128

    for alpha in alphas:
        for beta in betas:
            for lr in learning_rates:
                for seed in seeds:
                    log = 'logs/{dataset}/supervise/{encoder}.l-{layer}.{opt}-{lr}.a-{alpha}.b-{beta}.s-{seed}.txt'.format(
                        dataset=dataset,
                        encoder=encoder,
                        layer=layer,
                        opt=opt,
                        lr=lr,
                        alpha=alpha,
                        beta=beta,
                        seed=seed
                    )

                    err = '{encoder}.l-{layer}.{opt}-{lr}.a-{alpha}.b-{beta}.err'.format(
                        encoder=encoder,
                        layer=layer,
                        opt=opt,
                        lr=lr,
                        alpha=alpha,
                        beta=beta,
                    )

                    command = 'python sl.py -d {dataset} --lr {lr} -e {encoder} --seed {seed}'.format(
                        dataset=dataset,
                        encoder=encoder,
                        layer=layer,
                        opt=opt,
                        lr=lr,
                        seed=seed
                    ) + ' --alpha {alpha} --beta {beta} -b {batch_size} > {log} 2> {err}'.format(
                        alpha=alpha,
                        beta=beta,
                        batch_size=batch_size,
                        log=log,
                        err=err
                    )
                    append(command, 'commands.txt')


def schedule_supervise_simple(opt=DF_OPT,
                              encoder=BERTMLP,
                              layers='4',
                              lr=ZERO,
                              dataset='fed',
                              seeds='1111',
                              batch_sizes='64'):
    learning_rates = [float(x) for x in lr.split(',')]
    seeds = [int(x) for x in seeds.split(',')]

    batch_sizes = [int(x) for x in batch_sizes.split(',')]
    layer = 4

    for b in batch_sizes:
        for lr in learning_rates:
            for seed in seeds:
                log = f'logs/{dataset}/supervise/{encoder}.l-{layer}.b-{b}.{opt}-{lr}.s-{seed}.txt'
                err = f'{encoder}.l-{layer}.{opt}-{lr}.err'
                command = f'python sl.py -d {dataset} --lr {lr} -e {encoder} --seed {seed} -b {b} > {log} 2> {err}'
                append(command, 'commands.txt')


if __name__ == '__main__':
    # schedule_supervise_simple(encoder='berted',
    #                           layers='12',
    #                           lr='0.0004,0.0002,0.0001,0.00005',
    #                           batch_sizes='64')

    schedule(encoder=BERTMLP, dataset='fed', xis='0.05,0.1')
    schedule(encoder=BERTGCN, dataset='fed', xis='0.05,0.1')
