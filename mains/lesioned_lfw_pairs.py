import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader
from experiments.lesioned_lfw_pairs import LesionedLFWPairsExperiment
from datasets.lfw_pairs import lfw_pairs_dataset
from file_utils import *


def lesioned_lfw_pairs_main(args, param, values):
    runs = 1
    for value in np.array(values.split(',')).astype(int):
        hparams = {param: value.item()}
        args[param] = value.item()
        print(hparams)

        for run in range(runs):
            print(f'run #{run + 1}')
            
            ten_folds_dataset = lfw_pairs_dataset(subset='10fold')

            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

            exp = LesionedLFWPairsExperiment(args)
            fold = 0
            for train_idx, val_idx in skf.split(ten_folds_dataset, ten_folds_dataset.targets):
                train_subset = Subset(ten_folds_dataset, train_idx)
                val_subset = Subset(ten_folds_dataset, val_idx)

                train_loader = DataLoader(train_subset, batch_size=int(args['bz']), num_workers=args['num_workers'],
                                          shuffle=True)
                val_loader = DataLoader(val_subset, batch_size=int(args['bz']), num_workers=args['num_workers'],
                                        shuffle=False)

                print('fold %d' % fold)

                exp.run(train_loader, val_loader)

                fold += 1

