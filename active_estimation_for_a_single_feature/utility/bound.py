import numpy as np
from scipy.stats import beta

def cp_low_bound(ones, zeros, delta=0.1):
    p_low, p_up = copper_interval(ones, zeros, delta)
    return min(p_low * (1 - p_low), p_up * (1 - p_up))


def bernstein_interval(ones, zeros, delta=0.1):
    n = (ones + zeros)
    p_hat = ones / n
    var_hat = p_hat * (1 - p_hat)
    epsilon1 = np.sqrt(2 * var_hat * (2/delta)) / n
    epsilon2 = 7 / 3 * np.log(2 / delta) / (n - 1)
    epsilon = epsilon1 + epsilon2
    return max(p_hat - epsilon, 0), min(p_hat + epsilon, 1)


def hoeffding_interval(ones, zeros, delta=0.1):
    n = (ones + zeros)
    if n != 0:
        p_hat = ones / n
        epsilon = np.sqrt(np.log(2 / delta) / 2 * n)
    else:
        return 0, 1
    return max(p_hat - epsilon, 0), min(p_hat + epsilon, 1)


def copper_interval(ones, zeros, delta=0.1):
    if ones + zeros == 0:
        return 0, 1
    elif ones == 0:
        p_low, p_up = 0, 1 - delta ** (1 / zeros)
    elif zeros == 0:
        p_low, p_up = delta ** (1 / ones), 1
    else:
        p_low, p_up = beta.ppf(delta / 2, ones, zeros + 1), beta.ppf(1 - delta / 2, ones + 1, zeros)

    return p_low, p_up


def entropy_bound(p_low, p_up):
    H = lambda p: np.sqrt(p * (1 - p)) * np.abs(np.log(p/(1-p))) if p != 0 and p != 1 else 0
    if p_low <= 0.08322217 <= p_up or p_low <= 1 - 0.08322217 <= p_up:
        return H(0.08322217)
    else:
        return max(H(p_low), H(p_up))


def var_bound(p_low, p_up):
    p = 1 / 2
    if p_up < 1 / 2:
        p = p_up
    elif p_low > 1 / 2:
        p = p_low

    return p * (1 - p)


