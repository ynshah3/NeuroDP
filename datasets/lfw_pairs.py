from torchvision.datasets import LFWPairs
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from matplotlib import pyplot as plt

class LFWPairsWithID(LFWPairs):
    def __init__(self, root: str, split: str = "10fold", image_set: str = "funneled", transform = None, target_transform = None, download: bool = False) -> None:
        super().__init__(root, split, image_set, transform, target_transform, download)
    
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image1, image2, target, pair_id) where target is `0` for different identities and `1` for the same identities, and pair_id is the unique identifier for the pair.
        """
        img1, img2, target = super().__getitem__(index)
        pair_id = index  # Use the index as a unique identifier
        return img1, img2, target, pair_id

def lfw_pairs_dataset(subset='train'):
    transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[0, 0, 0], std=[255.0, 255.0, 255.0]),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = LFWPairsWithID(root='./', split=subset, transform=transform, download=True)
    dataset.pair_indices = list(range(len(dataset)))  # Add pair indices to the dataset
    return dataset


if __name__ == '__main__':
    train_dataset = lfw_pairs_dataset(subset='train')
    test_dataset = lfw_pairs_dataset(subset='test')
    folds_dataset = lfw_pairs_dataset(subset='10fold')

    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    folds_loader = DataLoader(dataset=folds_dataset, batch_size=1, shuffle=False)

    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))
    folds_batch = next(iter(folds_loader))

    # Example usage
    plt.figure(figsize=(9, 6))
    for i in range(2):
        plt.subplot(2, 3, i*3 + 1)
        plt.imshow(train_batch[i].squeeze().permute(1, 2, 0))
        if i == 0:
            plt.title('{}'.format('same' if train_batch[2] == 1 else 'different'))
        plt.axis('off')
        plt.subplot(2, 3, i*3 + 2)
        plt.imshow(test_batch[i].squeeze().permute(1, 2, 0))
        if i == 0:
            plt.title('{}'.format('same' if test_batch[2] == 1 else 'different'))
        plt.axis('off')
        plt.subplot(2, 3, i*3 + 3)
        plt.imshow(folds_batch[i].squeeze().permute(1, 2, 0))
        if i == 0:
            plt.title('{}'.format('same' if folds_batch[2] == 1 else 'different'))
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('../assets/sample_lfw.png')
