import matplotlib.pyplot as plt
import numpy as np
from utility.utiliy_active import mean_confidence_interval


def plot(scores, data_name, number_of_features):
    zoom_from_index = 50
    mse = plt.subplot(211)
    mse_zoom = plt.subplot(212)
    plt.title(data_name + ' MSE Score')
    plt.xlabel('Samples')
    plt.ylabel('MI')

    for k, v in scores.items():
        mean, lb, ub = mean_confidence_interval(v)
        times = np.arange(len(mean))
        plt.subplot(mse)
        plt.fill_between(times, ub, lb, alpha=.3)
        plt.plot(times, mean, linewidth=1, label=k)
        plt.subplot(mse_zoom)
        plt.fill_between(times[zoom_from_index:], ub[zoom_from_index:], lb[zoom_from_index:], alpha=.3)
        plt.plot(times[zoom_from_index:], mean[zoom_from_index:], linewidth=1, label=k)

    plt.legend()
    plt.tight_layout()
    file_name = 'results/{}_k={}.png'.format(data_name, number_of_features)
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
