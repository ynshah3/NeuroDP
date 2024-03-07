from sklearn.datasets import fetch_lfw_pairs
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.transforms import functional as F

class LFWPairsDataset(Dataset):
    def __init__(self, pairs, labels, transform=None):
        self.pairs = pairs
        self.labels = labels
        self.transform = transform
        print("Pairs shape:", self.pairs.shape)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        label = self.labels[idx]
        pair = [F.to_pil_image(image) for image in pair]
        if self.transform:
            pair = [self.transform(image) for image in pair]
        image1, image2 = pair
        return image1, image2, label

def lfw_pairs_dataset(subset='train'):
    print("Fetching LFW pairs dataset subset '%s'" % subset)
    lfw_pairs = fetch_lfw_pairs(subset=subset, resize=1, color=True, funneled=True)
    print(f"shape of the dataset: {lfw_pairs.pairs.shape}")
    
    pairs = lfw_pairs.pairs
    labels = lfw_pairs.target
    
    transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32),
    v2.Normalize(mean=[0, 0, 0], std=[255.0, 255.0, 255.0]),
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
    for image1, image2, label in train_loader:
        print("Image 1 shape:", image1.shape)
        print("Image 2 shape:", image2.shape)
        print("Label:", label)
        break