import numpy as np, scipy.stats as st


def get_y_category(y):
    d = dict()

    for l in y:
        d[l] = l

    return sorted(d.keys())


def get_prob(x, y, samples):
    prob_x = {val: np.round(c / len(x), decimals=3) for val, c in zip(*np.unique(x, return_counts=True))}
    cont_x_know = {val: c for val, c in zip(*np.unique(x[samples], return_counts=True))}
    vals, counts = np.unique(list(zip(x[samples], y[samples])), return_counts=True, axis=0)
    prob_x_geven_y = {tuple(val): np.round(c / cont_x_know[val[0]], decimals=3) for val, c in zip(vals, counts)}
    return prob_x, prob_x_geven_y


def mean_confidence_interval(data, confidence=0.68):
    m, lb, ub = [], [], []
    for d in data.T:
        mean = np.mean(d)
        n = len(d)
        sem = np.std(d) / np.sqrt(n)
        if sem:
            l, u = st.t.interval(confidence, n - 1, loc=mean, scale=sem)
        else:
            l, u = mean, mean

        m.append(mean)
        lb.append(l)
        ub.append(u)

    return m, lb, ub


def stochastic_conditional_entropy(X, y, samples):
    ans = []

    for i, x in enumerate(X.T):
        cond_entropy = 0
        if len(samples):
            prob_x, prob_x_geven_y = get_prob(x, y, samples)
            for x_val, y_val in prob_x_geven_y.keys():
                p_xy = prob_x_geven_y[(x_val, y_val)]  # p_xy = P(Y= y|X = x)
                cond_entropy += -prob_x[x_val] * p_xy * np.log2(p_xy)
        ans.append((i, cond_entropy))

    return np.array(sorted(ans, key=lambda k: k[1]))

