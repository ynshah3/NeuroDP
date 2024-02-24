import torch
import torch.nn as nn
import numpy as np
from scipy.stats import kendalltau
from file_utils import log_to_file
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


class LesionedRetrainPlatesExperiment:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'\t\tusing {self.device}')

        self.model = get_model(map_location=self.device)
        self.num_features = self.model.module.decoder.linear.in_features
        self.head = self.model.module.decoder.linear.to(self.device).float()
        self.model.module.decoder.linear = nn.Identity()
        self.model.to(self.device).float()

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.head.parameters():
            param.requires_grad = True

        self.probe_epochs = int(args['probe_epochs'])
        self.lesion_iters = int(args['lesion_iters'])
        self.retrain_epochs = int(args['retrain_epochs'])
        self.region_idx = int(args['region_idx'])
        self.retrain_cortex = False if args['retrain_cortex'].lower() == 'false' else True
        print(f'\t\tretraining cortex: {self.retrain_cortex}')
        self.criterion = nn.CrossEntropyLoss()

    def get_healthy_rdm(self, loader):
        return get_rdm(self.model, loader, self.device)

    def run(self, loaders, path):
        metrics = {
            'loss_retrained': [],
            'bacc_retrained': [],
            'tau_lesioned': [],
            'tau_retrained': [],
            'p_lesioned': [],
            'p_retrained': []
        }
        regions = [self.model.module.V1, self.model.module.V2, self.model.module.V4, self.model.module.IT]
        region = regions[self.region_idx]

        rdm_healthy = get_rdm(self.model, loaders[2], self.device)
        log_to_file(path + '_rdm.txt', 'rdm=' + str(rdm_healthy.tolist()))

        conv_layers = [module for module in region.modules() if isinstance(module, torch.nn.Conv2d)]

        for i in range(self.lesion_iters):
            for param in self.model.parameters():
                param.requires_grad = False

            print(f'\t\t\tIteration {i + 1}')
            for x in conv_layers:
                prune.random_unstructured(x, name='weight', amount=0.2)

            torch.save(self.model.state_dict(), path + '_ckpt_pruned_' + str(i) + '.pt')

            rdm_pruned = get_rdm(self.model, loaders[2], self.device)
            log_to_file(path + '_rdm.txt', 'rdm_pruned_' + str(i) + '=' + str(rdm_pruned.tolist()))
            tau, p_value = kendalltau(rdm_healthy.flatten(), rdm_pruned.flatten())
            metrics['tau_lesioned'].append(tau)
            metrics['p_lesioned'].append(p_value)

            if self.args['retrain_cortex']:
                for param in self.model.parameters():
                    param.requires_grad = True
            else:
                for conv_layer in conv_layers:
                    for param in conv_layer.parameters():
                        param.requires_grad = True

            self.retrain(*loaders[3:])

            torch.save(self.model.state_dict(), path + '_ckpt_retrained_' + str(i) + '.pt')

            rdm_retrained = get_rdm(self.model, loaders[2], self.device)
            log_to_file(path + '_rdm.txt', 'rdm_retrained_' + str(i) + '=' + str(rdm_retrained.tolist()))
            tau, p_value = kendalltau(rdm_healthy.flatten(), rdm_retrained.flatten())
            metrics['tau_retrained'].append(tau)
            metrics['p_retrained'].append(p_value)

            loss, acc, _ = self.fit_probe(*loaders[:3])
            metrics['loss_retrained'].append(loss)
            metrics['bacc_retrained'].append(acc)
        return metrics

    def retrain(self, train_loader, val_loader):
        for param in self.head.parameters():
            param.requires_grad = True

        head_optimizer = torch.optim.SGD(
            list(filter(lambda p: p.requires_grad, self.head.parameters())),
            lr=self.args['lr_head'],
            momentum=0.9,
            weight_decay=self.args['weight_decay'],
        )

        print('\t\t\t\tRetraining...')

        self.train(self.head, train_loader, head_optimizer, verbose=True, iter_stop=self.retrain_epochs)
        # val_metrics = self.test(self.head, val_loader, is_val=True)
        # print(f'\t\t\t\tval_loss={val_metrics[0]:.4f}, val_bacc={val_metrics[1]:.4f}')

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

    def train(self, head, loader, optimizer, verbose=False, iter_stop=None):
        head.train()
        self.model.train()
        self.head.train()

        for it, (inputs, targets) in enumerate(loader):
            if it == iter_stop:
                break

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()

            loss, acc, _ = self.compute_metrics(head, inputs, targets, is_test=False)

            if verbose and (it % 20 == 0):
                print(f'\t\t\t\tIter {it + 1}: train_loss={loss:.4f}, train_bacc={acc:.4f}')
            
            loss.backward()
            optimizer.step()

    def test(self, head, loader, is_val=True):
        self.model.eval()
        head.eval()
        self.head.eval()

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
                        is_test=False,
                        hexes=hexes,
                        contrasts=contrasts,
                        hex_contrast_acc=hex_contrast_acc
                    )

                    test_loss += loss
                    test_bacc += b_acc

        return test_loss.detach().cpu().item() / len(loader), \
               test_bacc / len(loader.dataset), \
               hex_contrast_acc / 20
