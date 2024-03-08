from torchvision.datasets import LFWPairs
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from matplotlib import pyplot as plt


def lfw_pairs_dataset(subset='train'):
    transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[0, 0, 0], std=[255.0, 255.0, 255.0]),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = LFWPairs(root='/vision/group/LFW/', split=subset, transform=transform, download=True)

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
