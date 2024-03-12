import torch
import torch.nn as nn
from cornet_s import CORnet_S
import torch.nn.functional as F

def get_model(map_location=None):
    model_hash = '1d3f7974'
    model = CORnet_S()
    model = torch.nn.DataParallel(model)
    url = f'https://s3.amazonaws.com/cornet-models/cornet_s-{model_hash}.pth'
    ckpt_data = torch.hub.load_state_dict_from_url(url, map_location=map_location)
    model.load_state_dict(ckpt_data['state_dict'])
    return model

class HealthyLFWPairsExperiment:
    def __init__(self, args, finetune_loader):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'\t\tusing {self.device}')
        self.args = args
        
        self.model = get_model(map_location=self.device)
        
        # Replace the last linear layer with a linear probe
        self.num_features = self.model.module.decoder.linear.in_features
        self.model.module.decoder.linear = nn.Identity()
        self.model.to(self.device).float()

        self.finetune(finetune_loader)

        for param in self.model.parameters():
            param.requires_grad = False

        self.probe = nn.Linear(self.num_features*4, self.args['num_classes']).to(self.device).float()
        
        for param in self.probe.parameters():
            param.requires_grad = True
        
        self.epochs = int(args['epochs'])
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.SGD(
            # self.model.parameters where requires_grad is True and the classifier parameters
            list(filter(lambda p: p.requires_grad, self.probe.parameters())),
            lr=0.0001,
            momentum=0.9,
            weight_decay=args['weight_decay']
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=15)

    def finetune(self, finetune_loader):
        for param in self.model.parameters():
            param.requires_grad = True

        head = nn.Linear(self.num_features*4, self.args['num_classes'])
        head.to(self.device).float()

        optimizer = torch.optim.SGD(
            list(filter(lambda p: p.requires_grad, self.model.parameters())) + list(filter(lambda p: p.requires_grad, head.parameters())),
            lr=0.01,
            momentum=0.9,
            weight_decay=self.args['weight_decay'],
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=3)

        print('\t\tFinetuning...')
        self.model.train()
        head.train()

        for epoch in range(3):
            ft_loss = 0.
            ft_acc = 0.

            for images1, images2, targets, _ in finetune_loader:
                images1, images2, targets = images1.to(self.device), images2.to(self.device), targets.to(self.device).float()

                optimizer.zero_grad()

                loss, acc = self.compute_metrics(head, images1, images2, targets)
                ft_loss += loss
                ft_acc += acc.sum().item()

                loss.backward()
                optimizer.step()
            
            print(f'\t\t\tEpoch {epoch + 1}: train_loss={(ft_loss.detach().cpu().item() / len(finetune_loader)):.4f}, train_bacc={(ft_acc / len(finetune_loader.dataset)):.4f}')
            lr_scheduler.step()

    def run(self, train_loader, val_loader):
        metrics = {'val_loss': [], 'val_acc': [], 'pair_accs': [], 'pair_indices': []}
        self.model.eval()
        print('\t\tTraining probe')

        for epoch in range(self.epochs):
            self.train(train_loader)
            val_metrics = self.test(val_loader)
            metrics['val_loss'].append(val_metrics[0])
            metrics['val_acc'].append(val_metrics[1])
            metrics['pair_accs'].extend(val_metrics[2])
            metrics['pair_indices'].extend(val_metrics[3])
            print(f'\t\t\tEpoch {epoch + 1}: val_loss={val_metrics[0]:.4f}, val_acc={val_metrics[1]:.4f}')
            self.lr_scheduler.step()
        # test_metrics = self.test(test_loader)
        # print(f'\t\t\tTest loss: {test_metrics[0]}, Test Acc: {test_metrics[1]}')
        return metrics
    
    def compute_metrics(self, classifier, images1, images2, labels):
        features1 = self.model(images1)
        features2 = self.model(images2)
        
        diff = features1 - features2
        abs_diff = torch.abs(diff)
        sq_diff = diff ** 2
        sqrt_abs_diff = torch.sqrt(abs_diff)
        product = features1 * features2
        combined_features = torch.cat((product, abs_diff, sq_diff, sqrt_abs_diff), dim=1)
        
        similarity = classifier(combined_features).squeeze()
        
        loss = F.binary_cross_entropy_with_logits(similarity, labels)
        predicted = (F.sigmoid(similarity) > 0.5).long()
        accuracies = (predicted.squeeze() == labels).float()
        
        return loss, accuracies

    def train(self, loader):
        self.probe.train()
        for image1, image2, labels, _ in loader:
            image1 = image1.to(self.device)
            image2 = image2.to(self.device)
            labels = labels.to(self.device).float()
            
            self.optimizer.zero_grad()
            loss, _ = self.compute_metrics(self.probe, image1, image2, labels)
            loss.backward()
            self.optimizer.step()
    
    def test(self, loader):
        self.probe.eval()
        test_loss = 0.0
        test_acc = 0.0
        pair_accs = []
        pair_indices = []
        
        with torch.no_grad():
            for image1, image2, labels, indices in loader:
                image1 = image1.to(self.device)
                image2 = image2.to(self.device)
                labels = labels.to(self.device).float()
                
                loss, acc = self.compute_metrics(self.probe, image1, image2, labels)
                test_loss += loss
                test_acc += acc.sum().item()
                pair_accs.extend(acc.tolist())
                pair_indices.extend(indices.tolist())
        
        return test_loss.detach().cpu().item() / len(loader), test_acc / len(loader.dataset), pair_accs, pair_indices
