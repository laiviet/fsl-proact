import os


def one_precision(texts):
    fs = [float(x) for x in texts]
    fs = [round(x, 2) for x in fs]
    strings = ['{:.2f}'.format(x) for x in fs]
    return ' '.join(strings)


def extract_a_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    # print(path)
    model = 'df'
    dataset = 'df'
    encoder = 'df'
    N = None
    K = None
    alpha = 0.0
    beta = 0.0
    gamma = 0.0
    omega = 0.0
    xi = 0.0
    lr = 0.0

    tree = None

    dev_f1 = None
    test_f1 = None

    best_dev_f1 = 0.0
    best_test_f1 = 0.0
    bert_layer = 12

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 2:
            continue
        if parts[0] == 'dataset' or parts[0] == 'dataset:':
            dataset = parts[1]
        elif parts[0] == 'encoder' or parts[0] == 'encoder:':
            encoder = parts[1]
        elif parts[0] == 'model' or parts[0] == 'model:':
            model = parts[1]
        elif parts[0] == 'lr' or parts[0] == 'lr:':
            lr = float(parts[1])
        elif parts[0] == 'optimizer':
            optimizer = parts[1]
        elif parts[0] == 'way':
            N = parts[1]
        elif parts[0] == 'bert_layer':
            bert_layer = int(parts[1])
        elif parts[0] == 'shot':
            K = parts[1]
        elif parts[0] == 'alpha' or parts[0] == 'alpha:':
            alpha = float(parts[1])
        elif parts[0] == 'beta' or parts[0] == 'beta:':
            beta = float(parts[1])
        elif parts[0] == 'xi':
            xi = float(parts[1])
        elif parts[0] == 'tree':
            tree = parts[1]

        elif line.startswith('| @'):
            current_iter = int(parts[2])
            if current_iter > 6000:
                break
        elif line.startswith('-> dev'):
            dev_line = line
        elif line.startswith('-> test'):
            test_line = line
            dev_parts = dev_line.split()
            test_parts = test_line.split()
            if float(dev_parts[4]) >= best_dev_f1:
                best_dev_f1 = float(dev_parts[4])
                dev_f1 = dev_parts[-3:]
                test_f1 = test_parts[-3:]
            # if float(test_parts[4]) >= best_test_f1:
            #     best_test_f1 = float(test_parts[4])
            #     dev_f1 = dev_parts[-3:]
            #     test_f1 = test_parts[-3:]

    if dev_f1 != None:
        # alpha = '    -' if alpha == 0.0 else '{:5.0f}'.format(alpha)
        # beta  = '    -' if beta == 0.0 else '{:0.3f}'.format(beta)
        # gamma = '    -' if gamma == 0.0 else '{:7.2f}'.format(gamma)
        # omega = '    -' if omega == 0.0 else '{:5.1f}'.format(omega)

        # print(path)
        # print(dev_f1, test_f1)

        # if alpha == 0 and beta == 0:
        # if alpha > 0 and beta == 0:
        # if alpha == 0 and beta > 0:
        # if alpha > 0 and beta > 0:
        if True:
            # print(path)
            # print(dev_f1, test_f1)
            dev_f1 = one_precision(dev_f1)
            test_f1 = one_precision(test_f1)

            format = '{model:8s} {encoder:8s} {lr:.5f} {alpha:5.0f} {beta:.4f} {xi:.2f} {dev} {test}'
            # print(model, encoder, lr, alpha, beta, xi, dev_f1, test_f1)
            print(format.format(model=model[:8],
                                encoder=encoder,
                                alpha=alpha,
                                lr=lr,
                                beta=beta,
                                xi=xi,
                                dev=dev_f1,
                                test=test_f1))
    else:
        pass
        # print('Ignore: ', path)


if __name__ == '__main__':

    # Extract FSL
    base = './logs/fed/bertmlp'

    files = sorted(os.listdir(base))
    files = [os.path.join(base, x) for x in files]
    # files = [x for x in files if 'bertgc' in x]
    # files = [x for x in files if '10-10' in x]
    # files = [x for x in files if 's-1111' in x]
    for file in files:
        # print(file)
        extract_a_file(file)

