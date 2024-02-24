import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader
from experiments.lesioned_retrain_plates import LesionedRetrainPlatesExperiment
from datasets.plates import plates_dataset
from datasets.imagenet import imagenet_dataset
from file_utils import *


def lesioned_retrain_plates_main(args, param, values):
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

            train_val_dataset, test_dataset = plates_dataset('./datasets/plates/')
            train_val_labels = [label for _, label in train_val_dataset]
            imagenet_train, imagenet_val = imagenet_dataset('/vision/group/ImageNet_2012/')
            skf = StratifiedKFold(n_splits=5, shuffle=True)

            fold = 0
            for train_idx, val_idx in skf.split(train_val_dataset, train_val_labels):
                train_dataset = Subset(train_val_dataset, train_idx)
                val_dataset = Subset(train_val_dataset, val_idx)

                train_loader = DataLoader(train_dataset, batch_size=int(args['bz']), num_workers=args['num_workers'], shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=int(args['bz']), num_workers=args['num_workers'], shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=int(args['bz']), num_workers=args['num_workers'], shuffle=False)
                imgnet_train_loader = DataLoader(imagenet_train, batch_size=128, num_workers=args['num_workers'], shuffle=True)
                imgnet_val_loader = DataLoader(imagenet_val, batch_size=128, num_workers=args['num_workers'], shuffle=False)

                print('\t\tFold %d' % fold)

                exp = LesionedRetrainPlatesExperiment(args)
                metrics = exp.run(
                    [train_loader, val_loader, test_loader, imgnet_train_loader, imgnet_val_loader],
                    dpath + str(run) + '_' + str(fold)
                )

                for item in metrics.items():
                    fpath = dpath + str(run) + '_' + item[0] + '.txt'
                    log_to_file(fpath, ','.join(format(x, ".4f") for x in item[1]))

                fold += 1
                break
