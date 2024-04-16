import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader
from experiments.lesioned_plates import LesionedPlatesExperiment
from datasets.plates import plates_dataset
from visualize.hca import CONTRAST_VALUES, HEX_VALUES, visualize_hex_contrasts
from file_utils import *


def lesioned_plates_main(args, param, values):
    dpath_parent = '/vision/u/ynshah/NeuroDP/runs/' + args["name"] + '/' + param + '/'

    runs = args['runs']
    for value in np.array(values.split(',')).astype(int):
        hparams = {param: value.item()}
        args[param] = value.item()
        print(hparams)

        dpath = dpath_parent + str(value) + '/'
        os.makedirs(dpath, exist_ok=True)
        metrics_run = {i: np.zeros((runs, args['lesion_iters'], len(CONTRAST_VALUES), len(HEX_VALUES))) for i in range(4)}

        for run in range(runs):
            print(f'\trun #{run + 1}\n\t-------')

            train_val_dataset, test_dataset = plates_dataset('./datasets/plates/')
            train_val_labels = [label for _, label in train_val_dataset]
            skf = StratifiedKFold(n_splits=5, shuffle=True)
            metrics_fold = {i: np.zeros((args['lesion_iters'], len(CONTRAST_VALUES), len(HEX_VALUES))) for i in range(4)}

            for lesion_iter in range(30, 40):
                exp = LesionedPlatesExperiment(args, int(args['region_idx']), lesion_iter)
                fold = 0
                for train_idx, val_idx in skf.split(train_val_dataset, train_val_labels):
                    train_dataset = Subset(train_val_dataset, train_idx)
                    val_dataset = Subset(train_val_dataset, val_idx)

                    train_loader = DataLoader(train_dataset, batch_size=int(args['bz']), num_workers=args['num_workers'], shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=int(args['bz']), num_workers=args['num_workers'], shuffle=False)
                    test_loader = DataLoader(test_dataset, batch_size=int(args['bz']), num_workers=args['num_workers'], shuffle=False)

                    print('\t\tFold %d' % fold)

                    loss, acc, _ = exp.fit_probe(train_loader, val_loader, test_loader)
                    print(f'\t\t\t{loss},{acc}')
