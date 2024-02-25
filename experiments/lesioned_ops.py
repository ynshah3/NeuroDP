import torch
import torch.nn as nn
import numpy as np
from visualize.hca import HEX_VALUES, CONTRAST_VALUES
from visualize.rdm import get_rdm
import torch.nn.utils.prune as prune
from cornet_s import CORnet_S


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
        print(f'\t\tusing {self.device}')

        self.model = get_model(map_location=self.device)
        self.num_features = self.model.module.decoder.linear.in_features
        self.model.module.decoder.linear = nn.Identity()
        self.model.to(self.device).float()

        for param in self.model.parameters():
            param.requires_grad = False

        self.probe_epochs = int(args['probe_epochs'])
        self.lesion_iters = int(args['lesion_iters'])
        self.retrain_epochs = int(args['retrain_epochs'])
        self.criterion = nn.CrossEntropyLoss()

    def get_healthy_rdm(self, loader):
        return get_rdm(self.model, loader, self.device)

    def run(self, train_loader, val_loader, test_loader):
        metrics = {}

        for region_idx, region in enumerate(
                [self.model.module.V1, self.model.module.V2, self.model.module.V4, self.model.module.IT]):
            conv_layers = [module for module in region.modules() if isinstance(module, torch.nn.Conv2d)]
            conv_layer_orig = [x.weight.data.clone() for x in conv_layers]
            print(f"\t\tLesioning region {region_idx}")
            layer_metrics = {'loss_lesioned': [], 'bacc_lesioned': []}

            for i in range(self.lesion_iters):
                print(f'\t\t\tIteration {i + 1}')
                for x in conv_layers:
                    prune.random_unstructured(x, name='weight', amount=0.2)
                loss, acc = self.fit_probe(train_loader, val_loader, test_loader)
                layer_metrics['loss_lesioned'].append(loss)
                layer_metrics['bacc_lesioned'].append(acc)

            # reset current region
            for c, x in enumerate(conv_layers):
                prune.remove(x, 'weight')
                x.weight.data = conv_layer_orig[c].clone()

            metrics[region_idx] = layer_metrics

        return metrics

    def fit_probe(self, train_loader, val_loader, test_loader):
        probe = nn.Linear(self.num_features, self.args['num_classes'])
        probe.to(self.device).float()
        probe_optimizer = torch.optim.Adam(
            list(filter(lambda p: p.requires_grad, probe.parameters())),
            lr=self.args['lr'],
            weight_decay=self.args['weight_decay'],
        )
        probe_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=probe_optimizer,
                                                                        T_max=self.probe_epochs)
        print('\t\t\t\tFitting probe...')

        for epoch in range(self.probe_epochs):
            self.train(probe, train_loader, probe_optimizer)
            val_metrics = self.test(probe, val_loader, is_val=True)
            print(f'\t\t\t\tEpoch {epoch + 1}: val_loss={val_metrics[0]:.4f}, val_bacc={val_metrics[1]:.4f}')
            probe_lr_scheduler.step()

        loss, acc = self.test(probe, test_loader, is_val=False)

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

    def test(self, head, loader, is_val=True):
        self.model.eval()
        head.eval()

        test_loss = 0.0
        test_bacc = 0.0

        with torch.no_grad():
            if is_val:
                for inputs, targets in loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    loss, b_acc = self.compute_metrics(
                        head=head,
                        inputs=inputs,
                        targets=targets
                    )

                    test_loss += loss
                    test_bacc += b_acc
            else:
                for inputs, targets, _ in loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    loss, b_acc = self.compute_metrics(
                        head=head,
                        inputs=inputs,
                        targets=targets
                    )

                    test_loss += loss
                    test_bacc += b_acc

        return test_loss.detach().cpu().item() / len(loader), \
               test_bacc / len(loader.dataset)
