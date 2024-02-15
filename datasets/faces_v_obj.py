import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import matplotlib.pyplot as plt


def faces_v_obj_dataset(image_dir='./datasets/faces_v_obj/'):
    transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    train_dataset = ImageFolder(root=image_dir + 'train/', transform=transform)
    test_dataset = ImageFolder(root=image_dir + 'test/', transform=transform)

    return train_dataset, test_dataset


if __name__ == '__main__':
    train_dataset, test_dataset = faces_v_obj_dataset(image_dir='faces_v_obj/')
    train_loader = DataLoader(dataset=train_dataset, batch_size=5, num_workers=0, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=5, num_workers=0, shuffle=True)
    train_images, train_targets = next(iter(train_loader))
    test_images, test_targets = next(iter(test_loader))
    plt.figure(figsize=(9, 3))
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(train_images[i].permute(1, 2, 0))
        plt.title('T:{}'.format(train_targets[i]))
        plt.subplot(2, 5, 6 + i)
        plt.imshow(test_images[i].permute(1, 2, 0))
        plt.title('T:{}'.format(test_targets[i]))
    plt.tight_layout()
    plt.savefig('../assets/sample_faces_v_obj.png')
