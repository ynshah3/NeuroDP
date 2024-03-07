import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader
from datasets.lfw_people import lfw_people_dataset
from experiments.lesioned_lfw_people import LesionedLFWPeopleExperiment
from file_utils import *

def lesioned_lfw_people_main(args, param, values):
    dpath_parent = './runs/' + args["name"] + '/' + param + '/'
    runs = args['runs']
    for value in np.array(values.split(',')).astype(int):
        hparams = {param: value.item()}
        args[param] = value.item()
        print(hparams)
        dpath = dpath_parent + str(value) + '/'
        os.makedirs(dpath, exist_ok=True)
        for run in range(runs):
            print(f'\trun #{run + 1}\n\t-------')
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
                exp = LesionedLFWPeopleExperiment(args, n_classes)
                metrics = exp.run(train_loader, val_loader, test_loader)
                for item in metrics.items():
                    fpath_val_loss = dpath + str(run) + '_' + str(item[0]) + '_loss.txt'
                    fpath_val_bacc = dpath + str(run) + '_' + str(item[0]) + '_bacc.txt'
                    fpath_val_class_metrics = dpath + str(run) + '_' + str(item[0]) + '_class_metrics.txt'
                    log_to_file(fpath_val_loss, ','.join(format(x, ".4f") for x in item[1]['loss_lesioned']))
                    log_to_file(fpath_val_bacc, ','.join(format(x, ".4f") for x in item[1]['bacc_lesioned']))
                    log_to_file(fpath_val_class_metrics, str(item[1]['class_metrics_lesioned']))
                fold += 1
