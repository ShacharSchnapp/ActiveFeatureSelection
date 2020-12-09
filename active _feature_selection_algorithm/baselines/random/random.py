import numpy as np

from utility.utiliy_active import mi_score


def feature_ranking_random(X, y, k, prob_x, total_labels):
    n_samples = X.shape[0]
    random_samples = np.random.choice(np.arange(n_samples), total_labels, replace=False)
    score_random = np.zeros(total_labels)

    for i, rs in enumerate(random_samples):
        samples_i = random_samples[:i + 1]
        score_random[i] = mi_score(X, y, k, prob_x, samples_i)

    return score_random.tolist()