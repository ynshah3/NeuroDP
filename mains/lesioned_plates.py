import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader
from experiments.lesioned_plates import LesionedPlatesExperiment
from datasets.plates import plates_dataset
from file_utils import *


def lesioned_plates_main(args, param, values):
    runs = 1
    for value in np.array(values.split(',')).astype(int):
        hparams = {param: value.item()}
        args[param] = value.item()
        print(hparams)

        for run in range(runs):
            print(f'run #{run + 1}')
            
            train_val_dataset, test_dataset = plates_dataset('./datasets/plates/')
            train_val_labels = [label for _, label in train_val_dataset]
            skf = StratifiedKFold(n_splits=5, shuffle=True)

            exp = LesionedPlatesExperiment(args)
            fold = 0
            for train_idx, val_idx in skf.split(train_val_dataset, train_val_labels):
                train_dataset = Subset(train_val_dataset, train_idx)
                val_dataset = Subset(train_val_dataset, val_idx)

                train_loader = DataLoader(train_dataset, batch_size=int(args['bz']), num_workers=args['num_workers'], shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=int(args['bz']), num_workers=args['num_workers'], shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=int(args['bz']), num_workers=args['num_workers'], shuffle=False)

                print('fold %d' % fold)

                exp.run(train_loader, val_loader, test_loader)

                fold += 1

