import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from datasets.plates import plates_dataset


matplotlib.rc('font', **{'size': 15})


def get_rdm(model, loader, device):
    activation = {}

    def get_activation(name):
        def hook(_, __, output):
            activation[name] = output.detach()
        return hook

    gt_class = [loader.dataset[i][1] for i in range(len(loader.dataset))]
    lists = [[] for _ in range(10)]

    # sort indices of images into lists corresponding to same class
    for i in range(len(gt_class)):
        for j in range(10):
            if gt_class[i] == j:
                lists[j].append(i)

    model.avgpool.register_forward_hook(get_activation('penultimate_layer'))
    model.eval()

    activation_layers = dict()
    activation_layers[model.avgpool] = {}
    layer_name = model.avgpool

    for i in range(len(lists)):
        for j in range(120):
            img, label, _ = loader.dataset[lists[i][j]]
            img = img.to(device)
            model(img.unsqueeze(0))
            get_layer_activation = activation['penultimate_layer']
            get_layer_activation = get_layer_activation.cpu()
            activ = np.asarray(get_layer_activation).squeeze()
            activation_layers[layer_name]['img_' + str(i) + str(j)] = activ.flatten()

    df = pd.DataFrame(activation_layers[layer_name])
    return 1 - df.corr()


def visualize_rdm(matrix, save_path):
    f, ax = plt.subplots()
    plt.matshow(matrix, cmap='jet')
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=15)

    plt.xlabel('Visual Stimuli')
    plt.ylabel('Visual Stimuli')
    plt.clim(0, 1)

    plt.tick_params(
        axis='both',  # changes apply to the both
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        right=False,
        left=False,
        labelbottom=False,
        labeltop=False,
        labelleft=False
    )
    ax.set_xticks([], [])
    ax.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    train_dataset, test_dataset = plates_dataset(image_dir='../datasets/plates/')
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, num_workers=0, shuffle=True)
    rdm = get_rdm(model, test_loader, 'cpu')
    visualize_rdm(rdm, '../assets/sample_rdm.png')
