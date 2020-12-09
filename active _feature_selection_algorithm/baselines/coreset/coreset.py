import numpy as np
import random
import matplotlib.pyplot as plt
import math

from utility.utiliy import argmax
from utility.utiliy_active import mi_score


def distance(A, B):
    return np.sum(np.array(A) != np.array(B))


def distance_l2(A, B):
    return np.linalg.norm(np.array(A) - np.array(B))


def incremental_farthest_search(points, k, d=distance):
    remaining_points = points.tolist()
    random_start = np.random.random_integers(0, len(remaining_points) - 1)
    solution_idx = [random_start]
    solution_set = [remaining_points.pop(random_start)]
    distances = [d(p, solution_set[0]) for p in remaining_points]

    for _ in range(k - 1):
        y_idx = argmax(distances)
        distances.pop(y_idx)
        solution_set.append(remaining_points.pop(y_idx))
        solution_idx.append(np.random.choice(np.where(solution_set[-1] == points)[0]))
        for i, p in enumerate(remaining_points):
            distances[i] = min(distances[i], d(p, solution_set[-1]))

    return np.array(solution_idx)


def plot_points(points, ax=None, style={'marker': 'o', 'color': 'b'}, label=False):
    """plots a set of points, with optional arguments for axes and style"""
    if ax == None:
        ax = plt.gca()
    for ind, p in enumerate(points):
        ax.plot(p.real, p.imag, **style)
        if label:
            ax.text(p.real, p.imag, s=ind, horizontalalignment='center', verticalalignment='center')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)


def create_circle_points(n):
    """creates a list of n points on a circle of radius one"""
    return [math.cos(2 * math.pi * i / float(n)) + \
            1j * math.sin(2 * math.pi * i / float(n)) for i in range(n)]


def create_point_cloud(n):
    return [2 * random.random() - 1 + 1j * (2 * random.random() - 1) for _ in range(n)]


def test_circle():
    plt.figure(figsize=(5, 5))
    circle100 = np.array(create_circle_points(100))
    plot_points(circle100)
    coreset = incremental_farthest_search(circle100, 16, distance_l2)
    plot_points(circle100[coreset], style={'marker': 'o', 'color': 'r', 'markersize': 12}, label=True)
    plt.show()


def feature_ranking_coreset(X, y, k, prob_x, total_labels):
    coreset_samples = incremental_farthest_search(X, total_labels)
    coreset_score = np.zeros(total_labels)

    for i, rs in enumerate(coreset_samples):
        samples_i = coreset_samples[:i + 1]
        coreset_score[i] = mi_score(X, y, k, prob_x, samples_i)
    return coreset_score.tolist()

# test_circle()
# X, y = init_data('../data/musk_csv.mat')
# X = k_means_quantization(X, 10)
# data_idx = incremental_farthest_search(X, 500)
# print(len(data_idx))
