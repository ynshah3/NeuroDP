import torch
import torch.nn as nn
from torch.nn.utils import prune
from torch.utils.data import Subset, DataLoader
from cornet_s import CORnet_S
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict


def get_model(map_location=None):
    model_hash = '1d3f7974'
    model = CORnet_S()
    model = torch.nn.DataParallel(model)
    url = f'https://s3.amazonaws.com/cornet-models/cornet_s-{model_hash}.pth'
    ckpt_data = torch.hub.load_state_dict_from_url(url, map_location=map_location)
    model.load_state_dict(ckpt_data['state_dict'])
    return model


class LesionedLFWPairsExperiment:
    def __init__(self, args, finetune_loader):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'\t\tusing {self.device}')
        self.args = args

        self.model = get_model(map_location=self.device)

        # Replace the last linear layer with a linear probe
        self.num_features = self.model.module.decoder.linear.in_features
        self.model.module.decoder.linear = nn.Identity()
        self.model.to(self.device).float()

        self.finetune(finetune_loader)

        for param in self.model.parameters():
            param.requires_grad = False

        self.probe = nn.Linear(self.num_features * 4, self.args['num_classes']).to(self.device).float()

        for param in self.probe.parameters():
            param.requires_grad = True

        self.criterion = nn.BCELoss()
        self.probe_epochs = int(args['probe_epochs'])
        self.lesion_iters = int(args['lesion_iters'])
        self.region_idx = int(args['region_idx'])
        self.optimizer = torch.optim.SGD(
            # self.model.parameters where requires_grad is True and the classifier parameters
            list(filter(lambda p: p.requires_grad, self.probe.parameters())),
            lr=0.0001,
            momentum=0.9,
            weight_decay=args['weight_decay']
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=15)

    def finetune(self, finetune_loader):
        for param in self.model.parameters():
            param.requires_grad = True

        head = nn.Linear(self.num_features * 4, self.args['num_classes'])
        head.to(self.device).float()

        optimizer = torch.optim.SGD(
            list(filter(lambda p: p.requires_grad, self.model.parameters())) + list(
                filter(lambda p: p.requires_grad, head.parameters())),
            lr=0.01,
            momentum=0.9,
            weight_decay=self.args['weight_decay'],
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=3)

        print('\t\tFinetuning...')
        self.model.train()
        head.train()

        for epoch in range(3):
            ft_loss = 0.
            ft_acc = 0.

            for images1, images2, targets, indices in finetune_loader:
                images1, images2, targets = images1.to(self.device), images2.to(self.device), targets.to(
                    self.device).float()

                optimizer.zero_grad()

                loss, acc, _ = self.compute_metrics(head, images1, images2, targets, indices)
                ft_loss += loss
                ft_acc += acc.sum()

                loss.backward()
                optimizer.step()

            print(
                f'\t\t\tEpoch {epoch + 1}: train_loss={(ft_loss.detach().cpu().item() / len(finetune_loader)):.4f}, train_bacc={(ft_acc / len(finetune_loader.dataset)):.4f}')
            lr_scheduler.step()

    def run(self, dataset):
        metrics = {
            'loss': np.zeros(self.lesion_iters),
            'acc': np.zeros(self.lesion_iters),
            'per_class_acc': np.zeros((self.lesion_iters, 6000))
        }

        regions = [self.model.module.V1, self.model.module.V2, self.model.module.V4, self.model.module.IT]
        region = regions[self.region_idx]

        conv_layers = [module for module in region.modules() if isinstance(module, torch.nn.Conv2d)]

        for i in range(self.lesion_iters):
            for param in self.model.parameters():
                param.requires_grad = False

            print(f'\t\tIteration {i + 1}')
            for x in conv_layers:
                prune.random_unstructured(x, name='weight', amount=0.2)

            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

            for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, dataset.targets)):
                print(f'\t\t\tFold {fold + 1}')

                train_subset = Subset(dataset, train_idx)
                val_subset = Subset(dataset, val_idx)

                train_loader = DataLoader(train_subset, batch_size=int(self.args['bz']), num_workers=self.args['num_workers'],
                                          shuffle=True)
                val_loader = DataLoader(val_subset, batch_size=int(self.args['bz']), num_workers=self.args['num_workers'],
                                        shuffle=False)

                for epoch in range(self.probe_epochs):
                    self.train(train_loader)
                    print(f'\t\t\t\tFinished Epoch {epoch + 1}')
                loss, acc, per_class_acc = self.test(val_loader)
                print(f'\t\t\tTest loss: {loss}, Test Acc: {acc}')
                metrics['loss'][i] += loss
                metrics['acc'][i] += acc
                metrics['per_class_acc'][i] += per_class_acc

        metrics['loss'] /= 10.
        metrics['acc'] /= 10.
        metrics['per_class_acc'] /= 10.

        return metrics

    def compute_metrics(self, classifier, images1, images2, labels, indices, is_test=False):
        features1 = self.model(images1)
        features2 = self.model(images2)

        diff = features1 - features2
        abs_diff = torch.abs(diff)
        sq_diff = diff ** 2
        sqrt_abs_diff = torch.sqrt(abs_diff)
        product = features1 * features2
        combined_features = torch.cat((product, abs_diff, sq_diff, sqrt_abs_diff), dim=1)

        similarity = classifier(combined_features).squeeze()

        loss = F.binary_cross_entropy_with_logits(similarity, labels)
        predicted = (F.sigmoid(similarity) > 0.5).long()
        accuracies = (predicted.squeeze() == labels).float()

        pca = np.zeros(6000)
        for i in range(len(accuracies)):
            pca[indices[i]] = accuracies[i].detach().cpu()

        return loss, accuracies.sum(), pca

    def train(self, loader):
        self.probe.train()
        for image1, image2, labels, indices in loader:
            image1 = image1.to(self.device)
            image2 = image2.to(self.device)
            labels = labels.to(self.device).float()

            self.optimizer.zero_grad()
            loss, _, _ = self.compute_metrics(self.probe, image1, image2, labels, indices)
            loss.backward()
            self.optimizer.step()

    def test(self, loader):
        self.probe.eval()
        test_loss = 0.0
        test_acc = 0.0
        per_class_accs = np.zeros(6000)

        with torch.no_grad():
            for image1, image2, labels, indices in loader:
                image1 = image1.to(self.device)
                image2 = image2.to(self.device)
                labels = labels.to(self.device).float()

                loss, acc, pca = self.compute_metrics(self.probe, image1, image2, labels, indices, is_test=True)
                test_loss += loss
                test_acc += acc
                per_class_accs += pca

        return test_loss.detach().cpu().item() / len(loader), test_acc / len(loader.dataset), per_class_accs

