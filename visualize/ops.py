import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable


matplotlib.rc('font', **{'size': 15})


def visualize_ops_percent(ops_runs, save_path):
    ops_runs = np.array(ops_runs)

    plt.figure()
    mean = np.mean(ops_runs, axis=0)
    err = np.std(ops_runs, axis=0)
    x = np.arange(1, 51)
    plt.plot(x, mean, color='black')
    plt.fill_between(x, mean + err, mean - err, color='gray', alpha=0.5)
    plt.xlabel('Noise Percentage')
    plt.ylim(-0.1, 1.1)
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig(save_path + '.png', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    data = np.random.normal(size=(10, 50), loc=0.6, scale=0.05).clip(0.0, 1.0)
    visualize_ops_percent(data, '../assets/sample_ops_percent_plot')
