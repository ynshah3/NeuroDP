import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import v2


def imagenet_dataset(image_dir='/vision/group/ImageNet_2012/'):
    transform = v2.Compose([
        v2.RandomResizedCrop((224, 224)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[0., 0., 0.], std=[255., 255., 255.]),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(root=image_dir + 'train/', transform=transform)
    test_dataset = ImageFolder(root=image_dir + 'val/', transform=transform)

    return train_dataset, test_dataset


if __name__ == '__main__':
    train_data, test_data = imagenet_dataset()
    train_loader = DataLoader(dataset=train_data, batch_size=5, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=5, shuffle=False)
    train_batch = next(iter(train_loader))
    print('Images: {}, Targets: {}'.format(train_batch[0].shape, train_batch[1].shape))
    print('Random Train Targets: {}'.format(train_batch[1][:5]))
    test_batch = next(iter(test_loader))
    print('Images: {}, Targets: {}'.format(test_batch[0].shape, test_batch[1].shape))
    print('Random Test Targets: {}'.format(test_batch[1][:5]))

