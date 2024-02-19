import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from visualize.hca import HEX_VALUES, CONTRAST_VALUES
from visualize.rdm import get_rdm


class LesionedPlatesExperiment:
    def __init__(self, args):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'\t\tusing {self.device}')

        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.model.to(self.device).float()

        for param in self.model.parameters():
            param.requires_grad = False

        self.probe = nn.Linear(num_features, args['num_classes'])
        self.probe.to(self.device).float()

        self.probe_epochs = args['probe_epochs']
        self.lesion_iters = args['lesion_iters']
        self.retrain_epochs = args['retrain_epochs']
        self.criterion = nn.CrossEntropyLoss()
        self.probe_optimizer = torch.optim.Adam(
            list(filter(lambda p: p.requires_grad, self.model.parameters())),
            lr=args['lr'],
            weight_decay=args['weight_decay'],
            betas=(args['beta1'], args['beta2']),
            eps=args['eps']
        )
        self.model_optimizer = torch.optim.SGD(
            list(filter(lambda p: p.requires_grad, self.model.parameters())),
            lr=args['lr'],
            weight_decay=args['weight_decay'],
            momentum=0.9
        )

    def get_healthy_rdm(self, loader):
        return get_rdm(self.model, loader, self.device)

    def run(self, train_loader, val_loader, test_loader):
        conv_layers = [module for module in self.model.modules() if
                       isinstance(module, torch.nn.Conv2d) and module.kernel_size != (1, 1)]

        metrics = {}

        for layer_idx, conv_layer in enumerate(conv_layers):
            print(f"\t\tLesioning Layer {layer_idx}")
            lesioned_indices = set()
            layer_metrics = {'loss_lesioned': [], 'bacc_lesioned': []}

            for i in range(self.lesion_iters):
                print(f'\t\t\tIteration {i + 1}')
                lesioned_indices = self.lesion_weights(conv_layer, lesioned_indices)
                loss, acc, _ = self.probe(train_loader, val_loader, test_loader)
                layer_metrics['loss_lesioned'].append(loss)
                layer_metrics['bacc_lesioned'].append(acc)

            metrics[layer_idx] = layer_metrics

        return metrics

    def lesion_weights(self, conv_layer, lesioned_indices):
        # Get the current weight tensor
        weight = conv_layer.weight.data

        # Calculate the number of weights to remove
        num_weights_to_remove = int(weight.size(0) * 0.2)

        # Randomly select weights that were not lesioned in previous iterations
        indices_to_remove = torch.randperm(weight.size(0))
        indices_to_remove = [idx.item() for idx in indices_to_remove if
                             idx.item() not in lesioned_indices[conv_layer]]
        indices_to_remove = indices_to_remove[:num_weights_to_remove]

        # Update lesioned indices
        lesioned_indices[conv_layer].update(indices_to_remove)

        # Remove weights
        weight[indices_to_remove, :, :, :] = 0

        # Make lesioned weights and units untrainable
        conv_layer.weight[indices_to_remove, :, :, :].requires_grad = False

        print(f'\t\tLesioned {weight[indices_to_remove, :, :, :].size} parameters')

        return lesioned_indices

    def probe(self, train_loader, val_loader, test_loader):
        metrics = {'val_loss': [], 'val_bacc': []}
        probe_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.probe_optimizer,
                                                                        T_max=self.probe_epochs)
        print('\t\t\t\tFitting probe...')

        for epoch in range(self.probe_epochs):
            self.train(train_loader)
            val_metrics = self.test(val_loader, is_val=True)
            metrics['val_loss'].append(val_metrics[0])
            metrics['val_bacc'].append(val_metrics[1])
            print(f'\t\t\t\tEpoch {epoch + 1}: val_loss={val_metrics[0]:.4f}, val_bacc={val_metrics[1]:.4f}')

            probe_lr_scheduler.step()

        loss, acc, hca = self.test(test_loader, is_val=False)

        return loss, acc, hca

    def compute_metrics(self, inputs, targets, is_test=False, hexes=None, contrasts=None, hex_contrast_acc=None):
        output = self.model(inputs)
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

    def train(self, loader):
        self.probe.train()

        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.probe_optimizer.zero_grad()

            loss, _, _ = self.compute_metrics(inputs, targets, is_test=False)

            loss.backward()
            self.probe_optimizer.step()

    def test(self, loader, is_val=True):
        self.model.eval()

        test_loss = 0.0
        test_bacc = 0.0
        hex_contrast_acc = np.zeros((len(CONTRAST_VALUES), len(HEX_VALUES)))

        with torch.no_grad():
            if is_val:
                for inputs, targets in loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    loss, b_acc, _ = self.compute_metrics(
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