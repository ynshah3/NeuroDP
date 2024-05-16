import numpy as np
from torch.utils.data import DataLoader
from experiments.lesioned_retrain_plates import LesionedRetrainPlatesExperiment
from datasets.plates import plates_dataset
from datasets.imagenet import imagenet_dataset
from file_utils import *


def lesioned_retrain_plates_main(args, param, values):
    for value in np.array(values.split(',')):
        hparams = {param: value.item()}
        args[param] = value.item()
        print(hparams)

        imagenet_train, imagenet_val = imagenet_dataset('/vision/group/ImageNet_2012/')

        imgnet_train_loader = DataLoader(imagenet_train, batch_size=128, num_workers=8, shuffle=True)
        imgnet_val_loader = DataLoader(imagenet_val, batch_size=128, num_workers=8, shuffle=False)

        exp = LesionedRetrainPlatesExperiment(args)
        exp.run(imgnet_train_loader, imgnet_val_loader)

