import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader
from datasets.lfw import lfw_people_dataset
from experiments.healthy_lfw import HealthyLFWExperiment
from file_utils import *


def healthy_lfw_experiment(args, param, values):
    dpath_parent = './runs/' + args["name"] + '/' + param + '/'

    runs = args['runs']
    for value in np.array(values.split(',')).astype(float):
        hparams = {param: value.item()}
        args[param] = value.item()
        print(hparams)

        dpath = dpath_parent + str(value) + '/'
        os.makedirs(dpath, exist_ok=True)

        lfw_runs = []

        for run in range(runs):
            print(f'\trun #{run + 1}\n\t-------')

            fpath_val_loss = dpath + str(run) + '_val_loss.txt'
            fpath_val_bacc = dpath + str(run) + '_val_bacc.txt'
            fpath_val_class_metrics = dpath + str(run) + '_val_class_metrics.txt'
            min_faces_per_person = args['min_faces_per_person']
            train_val_dataset, test_dataset, n_classes = lfw_people_dataset(min_faces_per_person=min_faces_per_person)
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

                exp = HealthyLFWExperiment(args, n_classes)
                metrics, class_metrics = exp.run(train_loader, val_loader, test_loader)
                log_to_file(fpath_val_loss, ','.join(format(x, ".4f") for x in metrics['val_loss']))
                log_to_file(fpath_val_bacc, ','.join(format(x, ".4f") for x in metrics['val_bacc']))
                log_to_file(fpath_val_class_metrics, ','.join(format(x, ".4f") for x in class_metrics))
                fold += 1
            run /= 5.0
            lfw_runs.append(run)
