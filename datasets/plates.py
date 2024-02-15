import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import os


class PlatesTestDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.hex_colors = []
        self.contrasts = []
        self.parse_filenames()

    def parse_filenames(self):
        for filename in self.imgs:
            file_name = os.path.basename(filename[0])
            _, hex_color, contrast = os.path.splitext(file_name)[0].split('_')
            self.hex_colors.append(hex_color)
            self.contrasts.append(float(contrast))

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        return img, label, (self.hex_colors[idx], self.contrasts[idx])


def plates_dataset(image_dir='./datasets/plates/'):
    transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    train_dataset = ImageFolder(root=image_dir + 'train/', transform=transform)
    test_dataset = PlatesTestDataset(root=image_dir + 'test/', transform=transform)

    return train_dataset, test_dataset


if __name__ == '__main__':
    train_dataset, test_dataset = plates_dataset(image_dir='plates/')
    train_loader = DataLoader(dataset=train_dataset, batch_size=5, num_workers=0, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=5, num_workers=0, shuffle=True)
    train_images, train_targets = next(iter(train_loader))
    test_images, test_targets, (test_hexes, test_contrasts) = next(iter(test_loader))
    plt.figure(figsize=(9, 3))
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(train_images[i].permute(1, 2, 0))
        plt.title('T:{}'.format(train_targets[i]))
        plt.subplot(2, 5, 6 + i)
        plt.imshow(test_images[i].permute(1, 2, 0))
        plt.title('T:{}, H:{}, C:{}'.format(test_targets[i], test_hexes[i], test_contrasts[i]))
    plt.tight_layout()
    plt.savefig('../assets/sample_plates.png')
