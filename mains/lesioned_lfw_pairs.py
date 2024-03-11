import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from datasets.lfw_pairs import lfw_pairs_dataset
from experiments.lesioned_lfw_pairs import LesionedLFWPairsExperiment
from file_utils import *

# /opt/conda/envs/pytorch/bin/python main.py healthy_lfw_pairs lr 0.01,0.001
def lesioned_lfw_pairs_main(args, param, values):
    dpath_parent = './' + args["name"] + '/' + param + '/'
    runs = args['runs']

    for value in np.array(values.split(',')).astype(float):
        hparams = {param: value.item()}
        args[param] = value.item()
        print(hparams)
        dpath = dpath_parent + str(value) + '/'
        os.makedirs(dpath, exist_ok=True)

        ten_folds_dataset = lfw_pairs_dataset(subset='10fold')
        print(len(ten_folds_dataset))

        finetune_dataset = Subset(ten_folds_dataset, np.random.randint(0, len(ten_folds_dataset), 1000))
        finetune_loader = DataLoader(finetune_dataset, batch_size=32, num_workers=args['num_workers'], shuffle=True)

        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kfold.split(ten_folds_dataset, ten_folds_dataset.targets)):
            print(f'\tFold #{fold + 1}\n\t-------')
            fpath_val_loss = dpath + str(fold) + '_val_loss.txt'
            fpath_val_acc = dpath + str(fold) + '_val_acc.txt'
            fpath_pair_accs = dpath + str(fold) + '_pair_accs.txt'
            fpath_pair_indices = dpath + str(fold) + '_pair_indices.txt'
            fpath_lesion_stats = dpath + str(fold) + '_lesion_stats.txt'  # New file path for lesion stats

            train_subset = Subset(ten_folds_dataset, train_idx)
            val_subset = Subset(ten_folds_dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=int(args['bz']), num_workers=args['num_workers'], shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=int(args['bz']), num_workers=args['num_workers'], shuffle=False)

            exp = LesionedLFWPairsExperiment(args, finetune_loader)
            metrics = exp.run(train_loader, val_loader)

            log_to_file(fpath_val_loss, ','.join(format(x, ".4f") for x in metrics['val_loss']))
            log_to_file(fpath_val_acc, ','.join(format(x, ".4f") for x in metrics['val_acc']))
            log_to_file(fpath_pair_accs, ','.join(format(x, ".4f") for x in metrics['pair_accs']))
            log_to_file(fpath_pair_indices, ','.join(str(x) for x in metrics['pair_indices']))
            # Log the lesion_stats to file
            log_to_file(fpath_lesion_stats, str(metrics['lesion_stats']))
