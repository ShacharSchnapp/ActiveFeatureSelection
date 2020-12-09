from bias_mechanism.bias import BiasMechanism
from mab_unit import MABUnit
from utility.utiliy import argmax
from utility.utiliy_active import *


class AFS:
    def __init__(self, x, y, prob_x, k, delta=0.05):
        self.delta = delta
        self.X = x.astype(np.int)
        self.y = y
        self.n_samples, self.n_features = self.X.shape
        self.mab_units = []
        self.prob_x = prob_x
        self.k = k
        self.bias_mechanism = BiasMechanism(self.X)
        self.D = {}
        self.q_hat = {}
        self.cond_entropy_score = []
        self.top_k = []

    def get_context_score(self, i, idx, sample, pair_on_sample, agg_func):
        ans = []
        idx_not_i = idx[idx != i]
        for j in idx_not_i:
            t = tuple(np.sort([i, j]))
            t_x = (sample[t[0]], sample[t[1]])
            ans += [self.bias_mechanism.pair_all_data[t][t_x] / pair_on_sample[t].get(t_x, 1)]

        return agg_func(ans)

    def score_context(self, sample, score, idx, pair_on_sample, agg_func):
        context_score = {i: self.get_context_score(i, idx[idx != i], sample, pair_on_sample, agg_func) for i in
                         idx}

        return agg_func(np.array([context_score[i] * score[i][val] for i, val in zip(idx, sample[idx])]))

    def sum_score(self, sample, score, idx):
        return np.sum(np.array([score[i][val] for i, val in zip(idx, sample[idx])]))

    def init_afs(self):
        self.mab_units = [MABUnit(f, self.delta) for i, f in enumerate(self.X.T)]
        self.D = {i: {v: 0 for v in self.mab_units[i].f_val} for i in range(self.n_features)}
        self.q_hat = {i: {v: 0.5 for v in self.mab_units[i].f_val} for i in range(self.n_features)}
        self.cond_entropy_score = []

    def calc_score(self, idx):
        return {i: {val: self.mab_units[i].get_reward(val) for val in self.mab_units[i].val_list} for i in idx}

    def active_sample(self, active_sample, available_index, idx, agg_func):
        score = self.calc_score(idx)
        if agg_func:
            pair_on_sample = self.bias_mechanism.get_pair_on_sample(active_sample, idx)
            list_of_score = np.array([self.score_context(x, score, idx, pair_on_sample, agg_func)
                                      for x in self.X[available_index]])
        else:
            list_of_score = np.array([self.sum_score(x, score, idx) for x in self.X[available_index]])

        return argmax(list_of_score)

    def update_q_hat(self, sample, y, active_samples):
        for f, val in enumerate(sample):
            active_samples_gev_jv = [i_t for i_t in active_samples if self.X[i_t][f] == val]
            self.D[f][val] += y
            n_jv = len(active_samples_gev_jv)
            self.q_hat[f][val] = self.D[f][val] / n_jv

    def update_sample(self, sample, y, active_samples):
        self.update_q_hat(sample, y, active_samples)
        [self.mab_units[i].update_history(val, y) for i, val in enumerate(sample)]

    def get_index_for_score(self, samples):
        H = stochastic_conditional_entropy(self.X, self.y, samples, self.prob_x, self.q_hat)
        M = H[:self.k, 0]
        M_hat = []

        all_lb = []
        for i in range(self.n_features):
            h_l, h_u = self.mab_units[i].get_entropy_confidence_interval()
            all_lb.append(h_l)
            if i in M:
                M_hat.append((i, h_u))
            else:
                M_hat.append((i, h_l))

        M_hat = np.array(sorted(M_hat, key=lambda k: k[1]))
        self.cond_entropy_score.append(np.sum(H[:self.k, 1]))
        idx_sort = random_argsort(M_hat[:, 1])
        M_hat = np.array(M_hat)[idx_sort][:self.k, 0]
        idx = np.array([i for i in M_hat if i not in M] + [i for i in M if i not in M_hat], dtype=int)

        return idx

    def is_plato(self, size=30, epsilon=0.005):
        is_plato = False
        t = len(self.cond_entropy_score)
        if t > size:
            is_plato = True
            for i in range(t - size, t - 1):
                if self.cond_entropy_score[i - 1] != 0 and np.abs(
                        1 - self.cond_entropy_score[-1] / self.cond_entropy_score[i - 1]) > epsilon:
                    return False

        return is_plato

    def run(self, total_labels, method='widows', agg_func=np.mean, plato=30):
        self.init_afs()
        available_index = np.arange(self.n_samples)
        score_active = np.zeros(total_labels)
        active_samples = []
        random = False

        for i in range(total_labels):
            if plato and (random or self.is_plato(plato)):
                random = True
                bast_sample_index = np.random.choice(np.arange(len(available_index)))
            else:
                if method == 'widows':
                    idx = self.get_index_for_score(active_samples)
                elif method == 'single':
                    idx = np.random.choice(np.arange(self.n_features), 1)
                else:
                    idx = np.arange(self.n_features)

                if len(idx) == 0:
                    score_active[i] = mi_score(self.X, self.y, self.k, self.prob_x, active_samples)
                    continue

                bast_sample_index = self.active_sample(active_samples, available_index, idx, agg_func)

            active_sample = available_index[bast_sample_index]
            active_samples.append(active_sample)
            available_index = np.delete(available_index, bast_sample_index)

            score_active[i] = mi_score(self.X, self.y, self.k, self.prob_x, active_samples)
            self.update_sample(self.X[active_sample], self.y[active_sample], active_samples)

        H = stochastic_conditional_entropy(self.X, self.y, active_samples, self.prob_x, self.q_hat)
        self.top_k = H[:self.k, 0]
        return score_active
