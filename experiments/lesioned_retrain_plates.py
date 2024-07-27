import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader
from scipy.stats import kendalltau
from file_utils import log_to_file
from visualize.rdm import get_rdm
import torch.nn.utils.prune as prune
from cornet_s import CORnet_S
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm


def get_model(map_location=None):
    model_hash = '1d3f7974'
    model = CORnet_S()
    model = torch.nn.DataParallel(model)
    url = f'https://s3.amazonaws.com/cornet-models/cornet_s-{model_hash}.pth'
    ckpt_data = torch.hub.load_state_dict_from_url(url, map_location=map_location)
    model.load_state_dict(ckpt_data['state_dict'])
    return model


class LesionedRetrainPlatesExperiment:
    def __init__(self, args):
        self.region_idx = int(args['region_idx'])
        self.lesion_iters = int(args['lesion_iters'])
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'using {self.device}')

        model = nn.Sequential(*list(resnet18().children())[:-2])
        checkpoint = torch.load('resnet_trained.pt', map_location='cuda')
        model.load_state_dict(checkpoint, strict=False)

        self.classifier = nn.Linear(512, 1000).to(self.device).float()
        checkpoint = torch.load('classifier_trained.pt', map_location='cuda')
        self.classifier.load_state_dict(checkpoint, strict=False)

        for param in model.parameters():
            param.requires_grad = True

        self.model = model.to(self.device).float()

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD( 
            list(self.model.parameters()) + list(self.classifier.parameters()),
            lr=float(self.args['lr']),
            momentum=0.9,
            weight_decay=self.args['weight_decay'],
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                                        T_max=40)

    def get_healthy_rdm(self, loader):
        return get_rdm(self.model, loader, self.device)

    def run(self, train_loader, val_loader):
        print(self.model)
        layer = self.model[self.region_idx + 4]
        print(layer)
        conv_layers = [module for module in layer.modules() if isinstance(module, torch.nn.Conv2d)]
        print(conv_layers)

        for param in self.model.parameters():
            param.requires_grad = True

        for param in self.classifier.parameters():
            param.requires_grad = True

        _, acc1, acc5 = self.test(val_loader)
        print(f'{acc1:.4f},{acc5:.4f}')
        with open('resnet.txt', 'a') as f:
            f.write(f'{0},{acc1:.4f},{acc5:.4f}')

        for i in range(50):
            print(f'Epoch {i + 1}')

            pruned, total = 0., 0.
            for x in conv_layers:
                m = prune.random_unstructured(x, name='weight', amount=0.2)
                pruned += torch.sum(m.weight_mask == 0)
                total += torch.sum(m.weight_mask == 0) + torch.sum(m.weight_mask == 1)

            print(pruned / total)
            self.retrain(train_loader)

            _, acc1, acc5 = self.test(val_loader)
            print(f'{acc1:.4f},{acc5:.4f}')
            with open('resnet.txt', 'a') as f:
                f.write(f'{i+1},{acc1:.4f},{acc5:.4f}')

            self.scheduler.step()


    def retrain(self, train_loader):
        self.model.train()
        self.classifier.train()

        for param in self.model.parameters():
            param.requires_grad = True

        for param in self.classifier.parameters():
            param.requires_grad = True

        iterator = iter(train_loader)
        for it in range(4096):
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, targets = next(iterator)

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            loss, acc1, acc5 = self.compute_metrics(inputs, targets)
            
            if it % 200 == 0 or it == 4095:
                print(f'\t{it + 1},{loss.item():.4f},{acc1:.4f},{acc5:.4f}')

            # torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.0)

            loss.backward()
            self.optimizer.step()

    def fit_probe(self, train_loader, val_loader, test_loader):
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.head.parameters():
            param.requires_grad = False

        probe = nn.Linear(self.num_features, self.args['num_classes'])
        probe.to(self.device).float()
        probe_optimizer = torch.optim.Adam(
            list(filter(lambda p: p.requires_grad, probe.parameters())),
            lr=self.args['lr_probe'],
            weight_decay=self.args['weight_decay'],
        )
        probe_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=probe_optimizer,
                                                                        T_max=self.probe_epochs)
        print('\t\t\tFitting probe...')

        for epoch in range(self.probe_epochs):
            self.train(probe, train_loader, probe_optimizer)
            val_metrics = self.test(probe, val_loader, is_val=True)
            print(f'\t\t\t\tEpoch {epoch + 1}: val_loss={val_metrics[0]:.4f}, val_bacc={val_metrics[1]:.4f}')
            
            probe_lr_scheduler.step()

        loss, acc = self.test(probe, test_loader, is_val=False)

        return loss, acc

    def accuracy(self, output, target, topk=(1,5)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k)
            return res

    def compute_metrics(self, inputs, targets):
        feats = self.model(inputs)
        feats = nn.functional.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
        output = self.classifier(feats)
        loss = self.criterion(output, targets)
        top1, top5 = self.accuracy(output, targets, topk=(1, 5))

        return loss, top1.item(), top5.item()

    def train(self, head, loader, optimizer):
        head.train()
        self.model.eval()
        self.head.eval()

        for it, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()

            loss, acc = self.compute_metrics(head, inputs, targets)

            loss.backward()
            optimizer.step()

    def test(self, loader, is_val=True):
        self.model.eval()
        self.classifier.eval()

        test_loss = 0.0
        test_top1 = 0.0
        test_top5 = 0.0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                loss, top1, top5 = self.compute_metrics(
                        inputs=inputs,
                        targets=targets,
                )

                test_loss += loss
                test_top1 += top1
                test_top5 += top5

        return test_loss.detach().cpu().item() / len(loader), \
               test_top1 / len(loader.dataset),\
               test_top5 / len(loader.dataset)
