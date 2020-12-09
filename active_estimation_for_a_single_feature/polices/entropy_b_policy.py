from utility.bound import entropy_bound, bernstein_interval
from polices.policy import Policy
import numpy as np


class EntropyBPolicy(Policy):
    def __init__(self, delta=0.1):
        super().__init__('I-B')
        self.delta = delta

    def step(self, mab):
        score_for_machine = []
        times_of_pull = mab.get_number_of_pull_per_machine()

        for x in range(mab.k):
            ones, zeros = mab.history[x][1], mab.history[x][0]
            p_low, p_up = bernstein_interval(ones, zeros, self.delta)
            score = (mab.prob_x[x] * np.sqrt(entropy_bound(p_low, p_up))) / times_of_pull[x]
            score_for_machine += [score]

        score_for_machine = np.array(score_for_machine)
        return self.argmax(score_for_machine)
