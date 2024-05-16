import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from visualize.hca import HEX_VALUES, CONTRAST_VALUES
from visualize.rdm import get_rdm
import torch.nn.utils.prune as prune
from cornet_s import CORnet_S, get_custom_cornet_s


def get_model(map_location=None):
    model_hash = '1d3f7974'
    model = CORnet_S()
    model = torch.nn.DataParallel(model)
    url = f'https://s3.amazonaws.com/cornet-models/cornet_s-{model_hash}.pth'
    ckpt_data = torch.hub.load_state_dict_from_url(url, map_location=map_location)
    model.load_state_dict(ckpt_data['state_dict'])
    return model


class LesionedOpsExperiment:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'using {self.device}')
        self.model = None
        self.criterion = nn.CrossEntropyLoss()

    def run(self, train_loader, val_loader, test_loader):
        
        for region_idx in range(4):
            print(f'\tregion {region_idx}')
            for lesion_iter in [-1] + list(range(40)):
                print(f'\tLesion {lesion_iter + 1}')

                self.model = get_custom_cornet_s(region_idx, lesion_iter)
                self.model.to(self.device).float()

                for param in self.model.parameters():
                    param.requires_grad = False
                
                self.fit_probe(train_loader, val_loader, test_loader)
            
    def fit_probe(self, train_loader, val_loader, test_loader):
        probe = nn.Linear(self.model.decoder.linear.in_features, self.args['num_classes'])
        self.model.decoder.linear = nn.Identity()
        probe.to(self.device).float()
        probe_optimizer = torch.optim.Adam(
            list(filter(lambda p: p.requires_grad, probe.parameters())),
            lr=self.args['lr'],
            weight_decay=self.args['weight_decay'],
        )
        probe_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=probe_optimizer,
                                                                        T_max=15)
        for epoch in range(15):
            self.train(probe, train_loader, probe_optimizer)
            loss, acc = self.test(probe, val_loader)
            print(f'\t\tval loss: {loss:.4f}, val acc: {acc:.4f}')
            probe_lr_scheduler.step()

        loss, acc = self.test(probe, test_loader)
        print(f'\ttest loss: {loss:.4f}, test acc: {acc:.4f}')

        return loss, acc

    def compute_metrics(self, head, inputs, targets):
        features = self.model(inputs)
        output = head(features)
        loss = self.criterion(output, targets)
        predicted = torch.argmax(output, 1)
        bacc = (targets == predicted).sum()

        return loss, bacc

    def train(self, head, loader, optimizer):
        head.train()

        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()

            loss, _ = self.compute_metrics(head, inputs, targets)

            loss.backward()
            optimizer.step()

    def test(self, head, loader):
        self.model.eval()
        head.eval()

        test_loss = 0.0
        test_bacc = 0.0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                loss, b_acc = self.compute_metrics(head=head, inputs=inputs, targets=targets)
                    
                test_loss += loss
                test_bacc += b_acc

        return test_loss.detach().cpu().item() / len(loader), \
            test_bacc / len(loader.dataset)

