import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader
from experiments.lesioned_ops import LesionedOpsExperiment
from datasets.fragments import fragments_dataset
from file_utils import *


def lesioned_ops_main(args, param, values):
    dpath_parent = '/vision/u/ynshah/NeuroDP/runs/' + args["name"] + '/' + param + '/'

    runs = args['runs']
    for value in np.array(values.split(',')):
        hparams = {param: value.item()}
        args[param] = value.item()
        print(hparams)

        dpath = dpath_parent + str(value) + '/'
        os.makedirs(dpath, exist_ok=True)

        for run in range(runs):
            print(f'\trun #{run + 1}\n\t-------')

            train_val_dataset, test_dataset = fragments_dataset('./datasets/fragments/')
            train_val_labels = [label for _, label in train_val_dataset]
            skf = StratifiedKFold(n_splits=5, shuffle=True)

            fold = 0
            for train_idx, val_idx in skf.split(train_val_dataset, train_val_labels):
                train_dataset = Subset(train_val_dataset, train_idx)
                val_dataset = Subset(train_val_dataset, val_idx)

                train_loader = DataLoader(train_dataset, batch_size=int(args['bz']), num_workers=args['num_workers'], shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=int(args['bz']), num_workers=args['num_workers'], shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=int(args['bz']), num_workers=args['num_workers'], shuffle=False)

                print('\t\tFold %d' % fold)

                exp = LesionedOpsExperiment(args)
                metrics = exp.run(train_loader, val_loader, test_loader)

                for item in metrics.items():
                    fpath_val_loss = dpath + str(run) + '_' + str(item[0]) + '_loss.txt'
                    fpath_val_bacc = dpath + str(run) + '_' + str(item[0]) + '_bacc.txt'
                    log_to_file(fpath_val_loss, ','.join(format(x, ".4f") for x in item[1]['loss_lesioned']))
                    log_to_file(fpath_val_bacc, ','.join(format(x, ".4f") for x in item[1]['bacc_lesioned']))

                fold += 1
