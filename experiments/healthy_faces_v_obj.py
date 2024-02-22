import torch
import torch.nn as nn
# from torchvision.models import resnet18, ResNet18_Weights
from cornet_s import CORnet_S


def get_model(map_location=None):
    model_hash = '1d3f7974'
    model = CORnet_S()
    model = torch.nn.DataParallel(model)
    url = f'https://s3.amazonaws.com/cornet-models/cornet_s-{model_hash}.pth'
    ckpt_data = torch.hub.load_state_dict_from_url(url, map_location=map_location)
    model.load_state_dict(ckpt_data['state_dict'])
    return model


class HealthyFacesVObjExperiment:
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
            val_metrics = self.test(val_loader)
            metrics['val_loss'].append(val_metrics[0])
            metrics['val_bacc'].append(val_metrics[1])
            print(f'\t\t\tEpoch {epoch + 1}: val_loss={val_metrics[0]}, val_bacc={val_metrics[1]}')

            self.lr_scheduler.step()

        loss, acc = self.test(test_loader)
        print(f'\t\t\tTest loss: {loss:.4f}, Test Acc: {acc:.4f}')

        return metrics

    def compute_metrics(self, inputs, targets):
        output = self.model(inputs)
        loss = self.criterion(output, targets)
        predicted = torch.argmax(output, 1)
        correct = (predicted == targets).sum().item()
        return loss, correct / len(targets)

    def train(self, loader):
        self.model.train()

        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            loss, _ = self.compute_metrics(inputs, targets)

            loss.backward()
            self.optimizer.step()

    def test(self, loader):
        self.model.eval()

        test_loss = 0.0
        test_bacc = 0.0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                loss, bacc = self.compute_metrics(inputs, targets)
                test_loss += loss.detach().cpu().item()
                test_bacc += bacc

        return test_loss / len(loader), test_bacc / len(loader.dataset)
