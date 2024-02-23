import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets.plates import plates_dataset
from cornet_s import CORnet_S


matplotlib.rc('font', **{'size': 15})


def get_model(map_location=None):
    model_hash = '1d3f7974'
    model = CORnet_S()
    model = torch.nn.DataParallel(model)
    url = f'https://s3.amazonaws.com/cornet-models/cornet_s-{model_hash}.pth'
    ckpt_data = torch.hub.load_state_dict_from_url(url, map_location=map_location)
    model.load_state_dict(ckpt_data['state_dict'])
    return model


def get_rdm(model, loader, device):
    activations_list = []

    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device)
            activations = model(images)
            activations_list.append(activations)

    activations_tensor = torch.cat(activations_list, dim=0)
    return 1 - np.corrcoef(activations_tensor.cpu().numpy())


def visualize_rdm(matrix, save_path):
    f, ax = plt.subplots()
    plt.matshow(matrix, fignum=f.number)
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
    model = get_model(map_location='cpu')
    model.module.decoder.linear = torch.nn.Identity()
    test_dataset = plates_dataset(image_dir='../datasets/plates/')[1]
    test_loader = DataLoader(dataset=test_dataset, batch_size=8, num_workers=0, shuffle=False)
    rdm = get_rdm(model, test_loader, 'cpu')
    visualize_rdm(rdm, '../assets/sample_rdm.png')
