import numpy as np
from utility.entropy_estimators import entropyd, entropyfromprobs
from utility.utiliy import binary_entropy
from utility.bound import copper_interval


class MABUnit:
    def __init__(self, f_val, delta):
        self.f_val = f_val
        self.entropy_x = entropyd(f_val)
        self.val_list, self.count = np.unique(f_val, return_counts=True)
        self.history = {}
        self.delta = delta
        for x in self.val_list:
            self.history[x] = {0: 0, 1: 0}
        self.t = 0
        self.prob_x = {x: c / len(self.f_val) for x, c in zip(self.val_list, self.count)}
        self.entropy_x = entropyfromprobs(self.prob_x.values())
        self.confidence_interval = {x: self.prob_confidence_interval(x) for x in self.val_list}
        self.score_function = None
        self.H = {x: self.calc_entropy_confidence_interval(x) for x in self.val_list}

    def prob_confidence_interval(self, x):
        return copper_interval(self.history[x][1], self.history[x][0], self.delta)

    def update_history(self, x, y):
        self.t += 1
        self.history[x][y] += 1
        self.confidence_interval[x] = self.prob_confidence_interval(x)
        self.H[x] = self.calc_entropy_confidence_interval(x)

    def get_entropy_confidence_interval(self):
        entropy = np.array([0.0, 0.0])
        for x, v in self.H.items():
            entropy += self.prob_x[x] * v
        return entropy

    def calc_entropy_confidence_interval(self, x):
        p_min, p_max = self.confidence_interval[x]
        lb = min(binary_entropy(p_min), binary_entropy(p_max))

        p = 1 / 2
        if p_min > 1 / 2:
            p = p_min
        elif p_max < 1 / 2:
            p = p_max

        ub = binary_entropy(p)

        return np.array([max(lb, 0), min(ub, 1)])

    def get_reward(self, x):
        return self.score(x) * self.prob_x[x]

    def score(self, x):
        times_of_pull = self.history[x][1] + self.history[x][0] + 1
        p_low, p_up = self.confidence_interval[x]
        H = lambda p: np.sqrt(p * (1 - p)) * np.abs(np.log(p / (1 - p))) if p != 0 and p != 1 else 0
        if p_low <= 0.08322217 <= p_up or p_low <= 1 - 0.08322217 <= p_up:
            return H(0.08322217) / times_of_pull
        else:
            return max(H(p_low), H(p_up)) / times_of_pull



