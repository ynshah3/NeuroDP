import torch
import torch.nn as nn
from cornet_s import CORnet_S
import tqdm
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
    def __init__(self, args):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'\t\tusing {self.device}')
        
        self.model = get_model(map_location=self.device)
        
        # Freeze the CORnet-S model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Replace the last linear layer with a linear probe
        num_features = self.model.module.decoder.linear.in_features
        self.model.module.decoder.linear = nn.Linear(num_features, 512)

        for param in self.model.module.decoder.linear.parameters():
            param.requires_grad = True
            
        self.model.to(self.device).float()
        
    # The classifier now handles concatenated features (1024) + abs diff (512) = 1536
        self.classifier = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        self.epochs = args['epochs']
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            # self.model.parameters where requires_grad is True and the classifier parameters
            list(filter(lambda p: p.requires_grad, self.model.parameters())) + list(self.classifier.parameters()),
            lr=args['lr'],
            weight_decay=args['weight_decay'],
            betas=(args['beta1'], args['beta2']),
            eps=args['eps']
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.epochs)
    
    def run(self, train_loader, val_loader, test_loader):
        metrics = {'val_loss': [], 'val_acc': []}
        for epoch in range(self.epochs):
            self.train(train_loader)
            val_metrics = self.test(val_loader)
            metrics['val_loss'].append(val_metrics[0])
            metrics['val_acc'].append(val_metrics[1])
            print(f'\t\t\tEpoch {epoch + 1}: val_loss={val_metrics[0]}, val_acc={val_metrics[1]}')
            self.lr_scheduler.step()
        # test_metrics = self.test(test_loader)
        # print(f'\t\t\tTest loss: {test_metrics[0]}, Test Acc: {test_metrics[1]}')
        return metrics
    
    def compute_metrics(self, image1, image2, labels):
        features1 = self.model(image1)
        features2 = self.model(image2)
        
        # Normalize features
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)
        
        diff = torch.abs(features1 - features2)
        combined_features = torch.cat([features1, features2, diff], dim=1) # Concatenate features
        
        similarity = self.classifier(combined_features)
        
        loss = self.criterion(similarity.squeeze(), labels)
        predicted = (similarity > 0.5).long()
        accuracy = (predicted.squeeze() == labels).float().mean()
        
        return loss, accuracy

    def train(self, loader):
        self.classifier.train()
        self.model.train()
        for image1, image2, labels in tqdm.tqdm(loader):
            image1 = image1.to(self.device)
            image2 = image2.to(self.device)
            labels = labels.to(self.device).float()
            
            self.optimizer.zero_grad()
            loss, _ = self.compute_metrics(image1, image2, labels)
            loss.backward()
            self.optimizer.step()
    
    def test(self, loader):
        self.classifier.eval()
        self.model.eval()
        test_loss = 0.0
        test_acc = 0.0
        
        with torch.no_grad():
            for image1, image2, labels in loader:
                image1 = image1.to(self.device)
                image2 = image2.to(self.device)
                labels = labels.to(self.device).float()
                
                loss, acc = self.compute_metrics(image1, image2, labels)
                test_loss += loss.item()
                test_acc += acc.item()
        
        return test_loss / len(loader), test_acc / len(loader)