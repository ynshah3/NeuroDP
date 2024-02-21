import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable


matplotlib.rc('font', **{'size': 15})

HEX_VALUES = ["ff0000", "ff4500", "ffa500", "ffd700", "ffff00", "adff2f", "008000", "00ff7f", "0000ff", "8a2be2",
              "7f00ff", "c71585"]

CONTRAST_VALUES = [1.050, 1.100, 1.200, 1.300, 1.500]
CONTRAST_NAMES = ['[1.001, 1.05)', '[1.05, 1.1)', '[1.1, 1.2)', '[1.2, 1.3)', '[1.3, 1.5)']


def visualize_hex_contrasts(hca_runs, save_path):
    hca_runs = np.array(hca_runs).transpose((1, 0, 2))
    markers = ['o', '^', 's', '+', '*']
    linestyles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1, 1, 1))]
    colors = ['darkgoldenrod', 'green', 'firebrick', 'dodgerblue', 'purple']

    num_colors = len(HEX_VALUES)
    color_wheel = [mcolors.hex2color('#' + hex_val) for hex_val in HEX_VALUES]

    # Create a custom colormap using the color wheel
    cmap = mcolors.ListedColormap(color_wheel)
    bounds = np.arange(num_colors + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot each curve for different contrast levels
    for i, contrast_value in enumerate(CONTRAST_NAMES):
        # Create a figure and axis
        fig, ax = plt.subplots()

        flierprops = dict(marker=markers[i], linestyle=linestyles[i], color=colors[i])
        medianprops = dict(linewidth=1.5, color=colors[i])
        plt.boxplot(hca_runs[i], flierprops=flierprops, medianprops=medianprops)

        colorbar = ax.figure.colorbar(ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation='horizontal',
                fraction=0.1, pad=0.02)
        colorbar.set_ticks(np.arange(num_colors) + 0.5)
        colorbar.set_ticklabels(['R', 'RO', 'O', 'YO', 'Y', 'YG', 'G', 'BG', 'B', 'BV', 'V', 'RV'])
        plt.xticks([])
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.ylabel('Accuracy')

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        plt.tight_layout()
        plt.savefig(save_path + '_' + str(i) + '.png', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    c1 = np.random.normal(size=(10, 12), loc=1, scale=1.00)
    c2 = np.random.normal(size=(10, 12), loc=4, scale=0.5)
    c3 = np.random.normal(size=(10, 12), loc=8, scale=1.75)
    c4 = np.random.normal(size=(10, 12), loc=15, scale=1.5)
    c5 = np.random.normal(size=(10, 12), loc=20, scale=0.75)
    data = np.array([c1, c2, c3, c4, c5]).transpose((1, 0, 2))
    visualize_hex_contrasts(data, 'assets/sample_hex_contrast_plot.png')
