from sklearn.datasets import fetch_lfw_pairs
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

class LFWPairsDataset(Dataset):
    def __init__(self, pairs, labels, transform=None):
        self.pairs = pairs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        label = self.labels[idx]
        
        if self.transform:
            pair = [self.transform(image) for image in pair]
        
        return pair, label

def lfw_pairs_dataset(subset='train', resize=0.5, color=True):
    lfw_pairs = fetch_lfw_pairs(subset=subset, resize=resize, color=color)
    
    pairs = lfw_pairs.pairs
    labels = lfw_pairs.target
    
    transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = LFWPairsDataset(pairs, labels, transform=transform)
    
    return dataset

if __name__ == '__main__':
    train_dataset = lfw_pairs_dataset(subset='train')
    test_dataset = lfw_pairs_dataset(subset='test')
    folds_dataset = lfw_pairs_dataset(subset='10_folds')
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    folds_loader = DataLoader(dataset=folds_dataset, batch_size=32, shuffle=False)
    
    # Example usage
    for pair, label in train_loader:
        image1, image2 = pair
        print("Image 1 shape:", image1.shape)
        print("Image 2 shape:", image2.shape)
        print("Label:", label)
        break