import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
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


class LesionedPlatesExperiment:
    def __init__(self, args, region_idx, lesion_iter):
        print(region_idx, lesion_iter)
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'\t\tusing {self.device}')

        # self.model = get_model(map_location=self.device)
        
        model = CORnet_S()
        model = torch.nn.DataParallel(model)
        regions = [model.module.V1, model.module.V2, model.module.V4, model.module.IT]
        region = regions[region_idx]

        url = f'/vision/u/ynshah/NeuroDP/runs/lesioned_retrain_plates_final/region_idx/checkpoints/{region_idx}_{lesion_iter}_4096_ckpt.pt'
        ckpt_data = torch.load(url, map_location='cpu')

        conv_layers = [module for module in region.modules() if isinstance(module, torch.nn.Conv2d)]
        for x in conv_layers:
            prune.random_unstructured(x, name='weight', amount=0.2)

        self.model = model
        self.model.load_state_dict(ckpt_data)
        
        self.num_features = self.model.module.decoder.linear.in_features
        self.model.module.decoder.linear = nn.Identity()

        self.model = self.model.module
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

        for region_idx, region in enumerate([self.model.module.V1, self.model.module.V2, self.model.module.V4, self.model.module.IT]):
            conv_layers = [module for module in region.modules() if isinstance(module, torch.nn.Conv2d)]
            conv_layer_orig = [x.weight.data.clone() for x in conv_layers]
            print(f"\t\tLesioning region {region_idx}")
            layer_metrics = {'loss_lesioned': [], 'bacc_lesioned': [], 'hca_lesioned': []}

            for i in range(self.lesion_iters):
                print(f'\t\t\tIteration {i + 1}')
                for x in conv_layers:
                    prune.random_unstructured(x, name='weight', amount=0.2)
                loss, acc, hca = self.fit_probe(train_loader, val_loader, test_loader)
                layer_metrics['hca_lesioned'].append(hca)
                layer_metrics['loss_lesioned'].append(loss)
                layer_metrics['bacc_lesioned'].append(acc)
            
            # reset current region
            for c, x in enumerate(conv_layers):
                prune.remove(x, 'weight')
                x.weight.data = conv_layer_orig[c].clone()

            layer_metrics['hca_lesioned'] = np.array(layer_metrics['hca_lesioned'])
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

        loss, acc, hca = self.test(probe, test_loader, is_val=False)

        return loss, acc, hca

    def compute_metrics(self, head, inputs, targets, is_test=False, hexes=None, contrasts=None, hex_contrast_acc=None):
        features = self.model(inputs)
        output = head(features)
        loss = self.criterion(output, targets)
        predicted = torch.argmax(output, 1)
        bacc = (targets == predicted).sum()

        if is_test:
            for i in range(len(predicted)):
                contrast_index = CONTRAST_VALUES.index(contrasts[i])
                hex_index = HEX_VALUES.index(hexes[i])
                if predicted[i] == targets[i]:
                    hex_contrast_acc[contrast_index][hex_index] += 1

        return loss, bacc, hex_contrast_acc

    def train(self, head, loader, optimizer):
        head.train()

        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()

            loss, _, _ = self.compute_metrics(head, inputs, targets, is_test=False)

            loss.backward()
            optimizer.step()

    def test(self, head, loader, is_val=True):
        self.model.eval()
        head.eval()

        test_loss = 0.0
        test_bacc = 0.0
        hex_contrast_acc = np.zeros((len(CONTRAST_VALUES), len(HEX_VALUES)))

        with torch.no_grad():
            if is_val:
                for inputs, targets in loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    loss, b_acc, _ = self.compute_metrics(
                        head=head,
                        inputs=inputs,
                        targets=targets,
                        is_test=False
                    )

                    test_loss += loss
                    test_bacc += b_acc
            else:
                for inputs, targets, (hexes, contrasts) in loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    loss, b_acc, hex_contrast_acc = self.compute_metrics(
                        head=head,
                        inputs=inputs,
                        targets=targets,
                        is_test=True,
                        hexes=hexes,
                        contrasts=contrasts,
                        hex_contrast_acc=hex_contrast_acc
                    )

                    test_loss += loss
                    test_bacc += b_acc

        return test_loss.detach().cpu().item() / len(loader), \
            test_bacc / len(loader.dataset), \
            hex_contrast_acc / 20
