from utility.utiliy import binary_entropy
import numpy as np
import scipy.stats as st
from sklearn.metrics import mutual_info_score


def get_y_category(y):
    d = dict()

    for l in y:
        d[l] = l

    return sorted(d.keys())


def get_prob(x, y, samples):
    cont_x_know = {val: c for val, c in zip(*np.unique(x[samples], return_counts=True))}
    vals, counts = np.unique(list(zip(x[samples], y[samples])), return_counts=True, axis=0)
    prob_x_geven_y = {tuple(val): np.round(c / cont_x_know[val[0]], decimals=3) for val, c in zip(vals, counts)}
    return prob_x_geven_y


def mean_confidence_interval(data, scale=1):
    m, lb, ub = [], [], []
    for d in data.T:
        mean = np.mean(d)
        n = scale * len(d)
        sem = np.std(d) / np.sqrt(n)
        if sem:
            l, u = st.t.interval(0.68, n - 1, loc=mean, scale=sem)
        else:
            l, u = mean, mean

        m.append(mean)
        lb.append(l)
        ub.append(u)

    return m, lb, ub


def random_argsort(arr):
    arr = np.array(arr)
    ans = []
    i = 0
    while i < len(arr):
        idx = np.flatnonzero(arr == arr[i])
        np.random.shuffle(idx)
        ans += idx.tolist()
        i = np.max(idx) + 1

    return np.array(ans, dtype=np.int)


def stochastic_conditional_entropy(X, y, samples, prob_x, q_hat=None):
    ans = []
    for i, x in enumerate(X.T):
        cond_entropy = 0
        if len(samples):
            for x_val in prob_x[i]:
                if q_hat:
                    p_xy = q_hat[i][x_val]
                else:
                    prob_x_geven_y = get_prob(x, y, samples)
                    if (x_val, 1) in prob_x_geven_y:
                        p_xy = prob_x_geven_y[(x_val, 1)]
                    elif (x_val, 0) in prob_x_geven_y:
                        p_xy = 0
                    else:
                        p_xy = 0.5

                cond_entropy += prob_x[i][x_val] * binary_entropy(p_xy)

        ans.append((i, cond_entropy))

    sort = np.array(sorted(ans, key=lambda k: k[1]))
    return sort[random_argsort(sort[:, 1])]


def mi_score(X, y, k, prob_x, samples_idx):
    M = stochastic_conditional_entropy(X, y, samples_idx, prob_x)[:k, 0].astype(int)
    return np.sum([mutual_info_score(X.T[i], y) for i in M])

# X = np.array([[0]*10 + [1]*10]).T
# y = np.array([1]*5 + [0]*5 + [1]*5 + [0]*5)
# stochastic_conditional_entropy(X, y, np.arange(20))


