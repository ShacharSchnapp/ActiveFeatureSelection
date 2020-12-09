import numpy as np


class BiasMechanism:
    def __init__(self, X):
        self.X = X
        self.pair_all_data = {}
        self.triplets_all_data = {}

    def get_pair_on_sample(self, samples, idx):
        for i in idx:
            idx_not_i = idx[idx != i]
            for j in idx_not_i:
                t = tuple(np.sort([i, j]))
                if t not in self.pair_all_data:
                    data = np.array([self.X.T[t[0]], self.X.T[t[1]]]).T
                    uniques, counts = np.unique(data, return_counts=True, axis=0)
                    self.pair_all_data[t] = {tuple(v): c for v, c in zip(uniques, counts)}

        pair_on_sample = {}
        smaple_data = self.X[samples]
        for i in idx:
            idx_not_i = idx[idx != i]
            for j in idx_not_i:
                t = tuple(np.sort([i, j]))
                if len(samples):
                    data = np.array([smaple_data.T[t[0]], smaple_data.T[t[1]]]).T
                    uniques, counts = np.unique(data, return_counts=True, axis=0)
                    pair_on_sample[t] = {tuple(v): c for v, c in zip(uniques, counts)}
                else:
                    pair_on_sample[t] = {}

        return pair_on_sample

    def get_triplets_on_sample(self, samples, idx):
        for i in idx:
            idx_not_i = idx[idx != i]
            for j in idx_not_i:
                idx_not_j = idx_not_i[idx_not_i != j]
                for r in idx_not_j:
                    t = tuple(np.sort([i, j, r]))
                    if t not in self.triplets_all_data:
                        data = np.array([self.X.T[t[0]], self.X.T[t[1]], self.X.T[t[2]]]).T
                        uniques, counts = np.unique(data, return_counts=True, axis=0)
                        self.triplets_all_data[t] = {tuple(v): c for v, c in zip(uniques, counts)}

        triplets_on_sample = {}
        sample_data = self.X[samples]
        for i in idx:
            idx_not_i = idx[idx != i]
            for j in idx_not_i:
                idx_not_j = idx_not_i[idx_not_i != j]
                for r in idx_not_j:
                    t = tuple(np.sort([i, j, r]))
                    if len(samples):
                        data = np.array([sample_data.T[t[0]], sample_data.T[t[1]], sample_data.T[t[2]]]).T
                        uniques, counts = np.unique(data, return_counts=True, axis=0)
                        triplets_on_sample[t] = {tuple(v): c for v, c in zip(uniques, counts)}
                    else:
                        triplets_on_sample[t] = {}

        return triplets_on_sample
