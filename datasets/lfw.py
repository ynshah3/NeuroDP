from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F


class LFWPeopleDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = F.to_pil_image(image)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def lfw_people_dataset(min_faces_per_person=50):
    lfw_people = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=0.4, color=True, funneled=True)
    X = lfw_people.images
    y = lfw_people.target
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32),
    v2.Normalize(mean=[0, 0, 0], std=[255.0, 255.0, 255.0]),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = LFWPeopleDataset(X_train, y_train, transform=transform)
    test_dataset = LFWPeopleDataset(X_test, y_test, transform=transform)

    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]
    return train_dataset, test_dataset, n_classes

if __name__ == '__main__':
    train_dataset, test_dataset, n_classes = lfw_people_dataset(min_faces_per_person=50)
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
    plt.savefig('../assets/sample_lfw.png')
