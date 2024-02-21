import torch
import torch.nn as nn
import numpy as np
# from torchvision.models import resnet18, ResNet18_Weights
from visualize.hca import HEX_VALUES, CONTRAST_VALUES
from cornet_s import CORnet_S


def get_model(map_location=None):
    model_hash = '1d3f7974'
    model = CORnet_S()
    model = torch.nn.DataParallel(model)
    url = f'https://s3.amazonaws.com/cornet-models/cornet_s-{model_hash}.pth'
    ckpt_data = torch.hub.load_state_dict_from_url(url, map_location=map_location)
    model.load_state_dict(ckpt_data['state_dict'])
    return model


class HealthyPlatesExperiment:
    def __init__(self, args):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'\t\tusing {self.device}')

        # self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model = get_model(map_location=self.device)

        # freeze resnet18 body
        for param in self.model.parameters():
            param.requires_grad = False

        num_features = self.model.module.decoder.linear.in_features
        self.model.module.decoder.linear = nn.Linear(num_features, args['num_classes'])

        # train linear probe
        for param in self.model.module.decoder.linear.parameters():
            param.requires_grad = True

        self.model.to(self.device).float()

        self.epochs = args['epochs']
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            list(filter(lambda p: p.requires_grad, self.model.parameters())),
            lr=args['lr'],
            weight_decay=args['weight_decay'],
            betas=(args['beta1'], args['beta2']),
            eps=args['eps']
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.epochs)

    def run(self, train_loader, val_loader, test_loader):
        metrics = {'val_loss': [], 'val_bacc': []}

        for epoch in range(self.epochs):
            self.train(train_loader)
            val_metrics = self.test(val_loader, is_val=True)
            metrics['val_loss'].append(val_metrics[0])
            metrics['val_bacc'].append(val_metrics[1])
            print(f'\t\t\tEpoch {epoch + 1}: val_loss={val_metrics[0]}, val_bacc={val_metrics[1]}')

            self.lr_scheduler.step()

        loss, acc, hex_contrast_acc = self.test(test_loader, is_val=False)
        print(f'\t\t\tTest loss: {loss:.4f}, Test Acc: {acc:.4f}')

        return metrics, hex_contrast_acc

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
        self.model.train()

        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            loss, _, _ = self.compute_metrics(inputs, targets, is_test=False)

            loss.backward()
            self.optimizer.step()

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
