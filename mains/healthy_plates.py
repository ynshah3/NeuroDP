import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader
from experiments.healthy_plates import HealthyPlatesExperiment
from datasets.plates import plates_dataset
from file_utils import *
from visualize import CONTRAST_VALUES, HEX_VALUES, visualize_hex_contrasts


def healthy_plates_main(args, param, values):
    dpath_parent = '/vision/u/ynshah/NeuroDP/runs/' + args["name"] + '/' + param + '/'

    runs = args['runs']
    for value in np.array(values.split(',')).astype(float):
        hparams = {param: value.item()}
        args[param] = value.item()
        print(hparams)

        dpath = dpath_parent + str(value) + '/'
        os.makedirs(dpath, exist_ok=True)

        hca_runs = []

        for run in range(runs):
            print(f'\trun #{run + 1}\n\t-------')

            fpath_val_loss = dpath + str(run) + '_val_loss.txt'
            fpath_val_bacc = dpath + str(run) + '_val_bacc.txt'

            hca_run = np.zeros((len(CONTRAST_VALUES), len(HEX_VALUES)))

            train_val_dataset, test_dataset = plates_dataset('./datasets/plates/')
            train_val_labels = [label for _, label in train_val_dataset]
            skf = StratifiedKFold(n_splits=5, shuffle=True)

            fold = 0
            for train_idx, val_idx in skf.split(train_val_dataset, train_val_labels):
                train_dataset = Subset(train_val_dataset, train_idx)
                val_dataset = Subset(train_val_dataset, val_idx)

                train_loader = DataLoader(train_dataset, batch_size=args['bz'], num_workers=args['num_workers'], shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=args['bz'], num_workers=args['num_workers'], shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=args['bz'], num_workers=args['num_workers'], shuffle=False)

                print('\t\tFold %d' % fold)

                exp = HealthyPlatesExperiment(args)
                metrics, hca = exp.run(train_loader, val_loader, test_loader)

                hca_run += hca

                log_to_file(fpath_val_loss, ','.join(format(x, ".4f") for x in metrics['val_loss']))
                log_to_file(fpath_val_bacc, ','.join(format(x, ".4f") for x in metrics['val_bacc']))

                fold += 1

            hca_run /= 5.0
            hca_runs.append(hca_run)

        visualize_hex_contrasts(hca_runs, dpath + 'hca')
