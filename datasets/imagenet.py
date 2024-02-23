import torch
from torchvision.datasets import ImageNet
from torchvision.transforms import v2


def imagenet_dataset(image_dir='/vision/group/ImageNet_2012/'):
    transform = v2.Compose([
        v2.RandomCrop((224, 224)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageNet(root=image_dir, split='train', transform=transform)
    test_dataset = ImageNet(root=image_dir, split='val', transform=transform)

    return train_dataset, test_dataset

