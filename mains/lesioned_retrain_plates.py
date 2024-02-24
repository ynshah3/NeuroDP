import numpy as np
from torch.utils.data import DataLoader
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
            imagenet_train, imagenet_val = imagenet_dataset('/vision/group/ImageNet_2012/')

            imgnet_train_loader = DataLoader(imagenet_train, batch_size=256, num_workers=args['num_workers'], shuffle=True)

            exp = LesionedRetrainPlatesExperiment(args)
            metrics = exp.run(
                [train_val_dataset, test_dataset, imgnet_train_loader],
                dpath + str(run)
            )

            for item in metrics.items():
                fpath = dpath + str(run) + '_' + item[0] + '.txt'
                log_to_file(fpath, ','.join(format(x, ".4f") for x in item[1]))
