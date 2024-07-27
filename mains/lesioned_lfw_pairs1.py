import numpy as np
from torch.utils.data import DataLoader, Subset
from datasets.lfw_pairs import lfw_pairs_dataset
from experiments.lesioned_lfw_pairs1 import LesionedLFWPairsExperiment
from file_utils import *


def lesioned_lfw_pairs_main(args, param, values):
    dpath_parent = './runs/' + args["name"] + '/' + param + '/'
    runs = args['runs']

    for value in np.array(values.split(',')):
        hparams = {param: value.item()}
        args[param] = value.item()
        print(hparams)

        dpath = dpath_parent + str(value) + '/'
        os.makedirs(dpath, exist_ok=True)

        for run in range(5):
            print(f'\trun #{run + 1}\n\t-------')

            ten_folds_dataset = lfw_pairs_dataset(subset='10fold')

            finetune_dataset = Subset(ten_folds_dataset, np.random.randint(0, len(ten_folds_dataset), 1000))
            finetune_loader = DataLoader(finetune_dataset, batch_size=32, num_workers=args['num_workers'], shuffle=True)

            exp = LesionedLFWPairsExperiment(args, finetune_loader)
            metrics = exp.run(ten_folds_dataset)
            print(metrics)

            fpath_loss = dpath + str(run) + '_loss.txt'
            fpath_bacc = dpath + str(run) + '_bacc.txt'
            fpath_per_class_acc = dpath + str(run) + '_pca.txt'
            log_to_file(fpath_loss, str(metrics['loss'].tolist()))
            log_to_file(fpath_bacc, str(metrics['acc'].tolist()))
            log_to_file(fpath_per_class_acc, str(metrics['per_class_acc'].tolist()))

