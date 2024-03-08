import numpy as np
from torch.utils.data import DataLoader, Subset
from datasets.lfw_pairs import lfw_pairs_dataset
from experiments.healthy_lfw_pairs import HealthyLFWPairsExperiment
from file_utils import *


def healthy_lfw_pairs_main(args, param, values):
    dpath_parent = './runs/' + args["name"] + '/' + param + '/'
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
            fpath_val_acc = dpath + str(run) + '_val_acc.txt'
            
            train_dataset = lfw_pairs_dataset(subset='train')
            test_dataset = lfw_pairs_dataset(subset='test')
            finetune_dataset = Subset(train_dataset, np.random.randint(0, len(train_dataset), 1000))
            
            train_loader = DataLoader(train_dataset, batch_size=int(args['bz']), num_workers=args['num_workers'], shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=int(args['bz']), num_workers=args['num_workers'], shuffle=False)
            finetune_loader = DataLoader(finetune_dataset, batch_size=128, num_workers=args['num_workers'], shuffle=True)
            
            exp = HealthyLFWPairsExperiment(args, finetune_loader)
            metrics = exp.run(train_loader, test_loader)
            log_to_file(fpath_val_loss, ','.join(format(x, ".4f") for x in metrics['val_loss']))
            log_to_file(fpath_val_acc, ','.join(format(x, ".4f") for x in metrics['val_acc']))
