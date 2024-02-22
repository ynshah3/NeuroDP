import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader
from experiments.healthy_faces_v_obj import HealthyFacesVObjExperiment
from datasets.faces_v_obj import faces_v_obj_dataset
from file_utils import *
from visualize.hca import CONTRAST_VALUES, HEX_VALUES, visualize_hex_contrasts


def healthy_faces_v_obj_main(args, param, values):
    dpath_parent = './NeuroDP/runs/' + args["name"] + '/' + param + '/'

    runs = args['runs']
    for value in np.array(values.split(',')).astype(float):
        hparams = {param: value.item()}
        args[param] = value.item()
        print(hparams)

        dpath = dpath_parent + str(value) + '/'
        os.makedirs(dpath, exist_ok=True)

        for run in range(runs):
            print(f'\trun #{run + 1}\n\t-------')

            fpath_val_loss = dpath + str(run) + '_val_loss.txt'
            fpath_val_bacc = dpath + str(run) + '_val_bacc.txt'

            train_val_dataset, test_dataset = faces_v_obj_dataset('./datasets/faces_v_obj/')
            num_classes = len(train_val_dataset.classes)
            class_names = train_val_dataset.classes
            class_to_idx = train_val_dataset.class_to_idx

            print("Number of classes:", num_classes)
            print("Class names:", class_names)
            print("Class to index mapping:", class_to_idx)

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

                exp = HealthyFacesVObjExperiment(args)
                metrics = exp.run(train_loader, val_loader, test_loader)
                log_to_file(fpath_val_loss, ','.join(format(x, ".4f") for x in metrics['val_loss']))
                log_to_file(fpath_val_bacc, ','.join(format(x, ".4f") for x in metrics['val_bacc']))

                fold += 1
