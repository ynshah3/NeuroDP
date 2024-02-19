import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from visualize.hca import HEX_VALUES, CONTRAST_VALUES
from visualize.rdm import get_rdm
import torch.nn.utils.prune as prune


class LesionedPlatesExperiment:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'\t\tusing {self.device}')

        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.model.to(self.device).float()

        self.probe_epochs = args['probe_epochs']
        self.lesion_iters = args['lesion_iters']
        self.retrain_epochs = args['retrain_epochs']
        self.criterion = nn.CrossEntropyLoss()
        # self.model_optimizer = torch.optim.SGD(
        #     list(filter(lambda p: p.requires_grad, self.model.parameters())),
        #     lr=args['lr'],
        #     weight_decay=args['weight_decay'],
        #    momentum=0.9
        # )

    def get_healthy_rdm(self, loader):
        return get_rdm(self.model, loader, self.device)

    def percent_pruned_params(self, module):
        pruned_params = torch.sum(module.weight_mask == 0)
        total = sum(param.numel() for param in module.parameters())
        return pruned_params / total

    def run(self, train_loader, val_loader, test_loader):
        conv_layers = [module for module in self.model.modules() if
                       isinstance(module, torch.nn.Conv2d) and module.kernel_size != (1, 1)]

        metrics = {}
        for layer_idx, conv_layer in enumerate(conv_layers):
            conv_layer_orig = conv_layer.weight.data.clone()
            print(f"\t\tLesioning Layer {layer_idx}")
            layer_metrics = {'loss_lesioned': [], 'bacc_lesioned': []}

            for i in range(self.lesion_iters):
                print(f'\t\t\tIteration {i + 1}')
                pruned_module = prune.random_unstructured(conv_layer, name='weight', amount=0.2)
                print(f"\t\t\tpruned params: {self.percent_pruned_params(pruned_module):.4f}%")
                loss, acc, _ = self.fit_probe(train_loader, val_loader, test_loader)
                layer_metrics['loss_lesioned'].append(loss)
                layer_metrics['bacc_lesioned'].append(acc)
            
            # reset current layer
            prune.remove(conv_layer, 'weight')
            conv_layer.weight.data = conv_layer_orig.clone()

            metrics[layer_idx] = layer_metrics

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
