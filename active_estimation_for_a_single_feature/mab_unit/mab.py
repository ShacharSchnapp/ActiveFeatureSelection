import numpy as np

from utility.utiliy import binary_entropy


class MAB:
    def __init__(self, porbs, prob_x, seed=None, times=2):
        self.k = len(porbs)
        self.porbs = np.array(porbs)
        self.n = 0
        self.history = {}
        self.seed = seed

        for x in range(self.k):
            self.history[x] = {0: 0, 1: 0}
        self.prob_x = prob_x
        for x in range(self.k):
            [self.pull_machine(x) for _ in range(times)]

    def pull_machine(self, x):
        if self.seed:
            random_int = self.seed[x][self.history[x][1] + self.history[x][0]]
        else:
            random_int = np.random.choice([0, 1], 1, p=[1 - self.porbs[x], self.porbs[x]])[0]

        self.history[x][random_int] += 1
        self.n += 1

    def beta_var(self, zeros, ones):
        a = zeros + 1
        b = ones + 1
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def beta_loss(self):
        return sum([self.beta_var(self.history[x][0], self.history[x][1]) for x in range(self.k)])

    def score_with_more_sample(self, x, y):
        self.history[x][y] += 1
        ans = self.beta_loss()
        self.history[x][y] -= 1
        return ans

    def get_p_hat(self):
        return np.array([self.history[x][1] / (self.history[x][1] + self.history[x][0]) for x in range(self.k)])

    def get_number_of_pull_per_machine(self):
        return np.array([self.history[x][1] + self.history[x][0] for x in range(self.k)])

    def get_sigma(self):
        return np.array([np.var([0] * self.history[x][0] + [1] * self.history[x][1]) for x in range(self.k)])

    def sigma_loss(self):
        return np.sum(self.get_sigma())

    def sigma_loss_with_more_sample(self, x, y):
        self.history[x][y] += 1
        sigma_loss = self.sigma_loss()
        self.history[x][y] -= 1
        return sigma_loss

    def mse_lost(self, p_hats):
        p_xy = np.sum(self.prob_x * self.porbs)
        p_xy_hat = np.sum(self.prob_x * p_hats)
        return np.abs(p_xy - p_xy_hat)
        # return np.linalg.norm(self.prob_x * (self.porbs - p_hats), 2)

    def entropy_lost(self, p_hats):
        H_hat = 0
        H = 0
        for i, p_hat in enumerate(p_hats):
            H_hat += self.prob_x[i] * binary_entropy(p_hat)
            H += self.prob_x[i] * binary_entropy(self.porbs[i])

        return np.abs(H_hat - H)
        # return np.power(H_hat - H, 2)

    def porb_hat_with_more_sample(self, x, y):
        self.history[x][y] += 1
        p_hat_x = self.get_p_hat()[x]
        self.history[x][y] -= 1
        return p_hat_x
