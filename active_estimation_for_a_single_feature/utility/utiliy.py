import numpy as np
import scipy
from sklearn.cluster import KMeans


def binary_entropy(p):
    if p == 0 or p == 1:
        return 0

    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def argmin(arr):
    return np.random.choice(np.flatnonzero(arr == arr.min()))


def argmax(arr):
    arr = np.array(arr)
    return np.random.choice(np.flatnonzero(arr == arr.max()))


def init_data(path):
    mat = scipy.io.loadmat(path)
    data = mat['X']  # data
    data = data.astype('float64')
    y = mat['Y']  # label
    y = y[:, 0]
    return data, y_to_two_label(y)


def y_to_two_label(y):
    vals, count = np.unique(y, return_counts=True)
    max_val = vals[np.argmax(np.unique(y, return_counts=True)[1])]
    return np.array(list(map(lambda x: 0 if x == max_val else 1, y)))


def remove_single_val_feature(X):
    mask = [len(np.unique(f)) > 1 for f in X.T]
    return X.T[mask].T


def k_means_quantization(X, n_clusters=10):
    print('quantization data in to', n_clusters, 'bins')
    data = np.zeros(X.shape)
    for i, f_val in enumerate(X.T):
        data[:, i] = f_val
        if len(np.unique(f_val)) > n_clusters:
            f_val = f_val.reshape(-1, 1)
            kmeans = KMeans(n_clusters=n_clusters, max_iter=1000).fit(f_val)
            data[:, i] = kmeans.predict(f_val)

    return data.astype(int)
