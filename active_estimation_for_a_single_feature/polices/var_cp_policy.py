import numpy as np
from polices.policy import Policy
from utility.bound import var_bound, copper_interval


class VARCPPolicy(Policy):
    def __init__(self, delta=0.05):
        super().__init__('VAR-CP')
        self.delta = delta

    def step(self, mab):
        times_of_pull = mab.get_number_of_pull_per_machine()
        score_for_machine = []

        for x in range(mab.k):
            ones, zeros = mab.history[x][1], mab.history[x][0]
            p_low, p_up = copper_interval(ones, zeros, self.delta)
            score = np.sqrt(var_bound(p_low, p_up)) / times_of_pull[x]
            score_for_machine += [mab.prob_x[x] * score]

        return self.argmax(np.array(score_for_machine))
