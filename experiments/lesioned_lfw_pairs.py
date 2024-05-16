import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
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


class LesionedLFWPairsExperiment:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'using {self.device}')
        self.model = None
        self.criterion = nn.BCELoss()

    def run(self, train_loader, val_loader):
        
        for region_idx in range(4):
            print(f'\tregion {region_idx}')
            for lesion_iter in [-1] + list(range(40)):
                print(f'\tLesion {lesion_iter + 1}')

                self.model = get_custom_cornet_s(region_idx, lesion_iter)
                self.model.to(self.device).float()

                for param in self.model.parameters():
                    param.requires_grad = False
                
                self.fit_probe(train_loader, val_loader)
            
    def fit_probe(self, train_loader, val_loader):
        probe = nn.Linear(self.model.decoder.linear.in_features * 4, self.args['num_classes'])
        self.model.decoder.linear = nn.Identity()
        probe.to(self.device).float()
        probe_optimizer = torch.optim.SGD(
            list(filter(lambda p: p.requires_grad, probe.parameters())),
            lr=self.args['lr'],
            momentum=0.9,
            weight_decay=self.args['weight_decay'],
        )
        probe_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=probe_optimizer,
                                                                        T_max=15)
        for epoch in range(15):
            loss, acc = self.train(probe, train_loader, probe_optimizer)
            print(f'\t\ttrain loss: {loss:.4f}, train acc: {acc:.4f}')
            probe_lr_scheduler.step()

        loss, acc = self.test(probe, val_loader)
        print(f'\ttest loss: {loss:.4f}, test acc: {acc:.4f}')

        return loss, acc

    def compute_metrics(self, head, images1, images2, labels):
        features1 = self.model(images1)
        features2 = self.model(images2)

        diff = features1 - features2
        abs_diff = torch.abs(diff)
        sq_diff = diff ** 2
        sqrt_abs_diff = torch.sqrt(abs_diff)
        product = features1 * features2
        combined_features = torch.cat((product, abs_diff, sq_diff, sqrt_abs_diff), dim=1)

        similarity = head(combined_features).squeeze()

        loss = F.binary_cross_entropy_with_logits(similarity, labels)
        predicted = (F.sigmoid(similarity) > 0.5).long()
        accuracies = (predicted.squeeze() == labels).float()

        return loss, accuracies.sum()

    def train(self, head, loader, optimizer):
        head.train()
        train_loss = 0.0
        train_acc = 0.0

        for images1, images2, targets in loader:
            images1, images2, targets = images1.to(self.device), images2.to(self.device), targets.to(self.device).float()

            optimizer.zero_grad()

            loss, acc = self.compute_metrics(head, images1, images2, targets)
            train_loss += loss
            train_acc += acc

            loss.backward()
            optimizer.step()

        return train_loss.detach().cpu().item() / len(loader), \
            train_acc / len(loader.dataset)

    def test(self, head, loader):
        self.model.eval()
        head.eval()

        test_loss = 0.0
        test_bacc = 0.0

        with torch.no_grad():
            for images1, images2, targets in loader:
                images1, images2, targets = images1.to(self.device), images2.to(self.device), targets.to(self.device).float()

                loss, b_acc = self.compute_metrics(head, images1, images2, targets)
                    
                test_loss += loss
                test_bacc += b_acc

        return test_loss.detach().cpu().item() / len(loader), \
            test_bacc / len(loader.dataset)

