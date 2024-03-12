import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from cornet_s import CORnet_S
from file_utils import log_to_file

def get_model(map_location=None):
    model_hash = '1d3f7974'
    model = CORnet_S()
    model = torch.nn.DataParallel(model)
    url = f'https://s3.amazonaws.com/cornet-models/cornet_s-{model_hash}.pth'
    ckpt_data = torch.hub.load_state_dict_from_url(url, map_location=map_location)
    model.load_state_dict(ckpt_data['state_dict'])
    return model

class LesionedLFWPairsExperiment:
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
            list(filter(lambda p: p.requires_grad, self.probe.parameters())),
            lr=0.0001,
            momentum=0.9,
            weight_decay=args['weight_decay']
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=15)

    def run_experiment(self, train_loader, val_loader):
        metrics = {}
        regions = [self.model.module.V1, self.model.module.V2, self.model.module.V4, self.model.module.IT]
        lesion_iters = int(self.args['lesion_iters'])
        for region_idx, region in enumerate(regions):
            print(f'\t\tLesioning region {region_idx}')
            conv_layers = [module for module in region.modules() if isinstance(module, torch.nn.Conv2d)]
            conv_layers_orig = [x.weight.data.clone() for x in conv_layers]
            layer_metrics = {'val_loss': [], 'val_acc': [], 'pair_accs': [], 'pair_indices': []}
            for i in range(lesion_iters):
                print(f'\t\t\tIteration {i + 1}')
                for lesioned_layer in conv_layers:
                    prune.random_unstructured(lesioned_layer, name='weight', amount=0.2)
                val_loss, val_acc, pair_accs, pair_indices = self.run(train_loader, val_loader)
                layer_metrics['val_loss'].append(val_loss)
                layer_metrics['val_acc'].append(val_acc)
                layer_metrics['pair_accs'].extend(pair_accs)
                layer_metrics['pair_indices'].extend(pair_indices)
                print(f'\t\t\t\tVal loss: {val_loss:.4f}, Val acc: {val_acc:.4f} at iteration {i + 1} of region {region_idx}') 
            
            for c, lesioned_layer in enumerate(conv_layers):
                prune.remove(lesioned_layer, 'weight')
                lesioned_layer.weight.data = conv_layers_orig[c].clone()
            
            layer_metrics['pair_accs'] = np.array(layer_metrics['pair_accs'])
            layer_metrics['pair_indices'] = np.array(layer_metrics['pair_indices'])
            layer_metrics['val_loss'] = np.array(layer_metrics['val_loss'])
            layer_metrics['val_acc'] = np.array(layer_metrics['val_acc'])
            metrics[region_idx] = layer_metrics
            # log to file per region
            fpath_val_loss = f'./{self.args["name"]}/val_loss_{region_idx}.txt'
            fpath_val_acc = f'./{self.args["name"]}/val_acc_{region_idx}.txt'
            fpath_pair_accs = f'./{self.args["name"]}/pair_accs_{region_idx}.txt'
            fpath_pair_indices = f'./{self.args["name"]}/pair_indices_{region_idx}.txt'
            log_to_file(fpath_val_loss, ','.join(format(x, ".4f") for x in layer_metrics['val_loss']))
            log_to_file(fpath_val_acc, ','.join(format(x, ".4f") for x in layer_metrics['val_acc']))
            log_to_file(fpath_pair_accs, ','.join(format(x, ".4f") for x in layer_metrics['pair_accs']))
            log_to_file(fpath_pair_indices, ','.join(str(x) for x in layer_metrics['pair_indices']))
        return metrics

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
        self.model.eval()
        print('\t\tTraining probe')

        for epoch in range(self.epochs):
            print(f'\t\t\tEpoch {epoch + 1}')
            self.train(train_loader)
            self.lr_scheduler.step()
        return self.test(val_loader)
    
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
        epoch_loss = 0.0
        epoch_acc = 0.0

        for image1, image2, labels, _ in loader:
            image1 = image1.to(self.device)
            image2 = image2.to(self.device)
            labels = labels.to(self.device).float()
            self.optimizer.zero_grad()
            loss, acc = self.compute_metrics(self.probe, image1, image2, labels)
            epoch_loss += loss
            epoch_acc += acc.sum().item()
            loss.backward()
            self.optimizer.step()
        
        print(f'\t\t\t\tTrain loss: {epoch_loss.detach().cpu().item() / len(loader):.4f}, Train acc: {epoch_acc / len(loader.dataset):.4f}')
    
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
