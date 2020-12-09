from polices.policy import Policy
import numpy as np


class ProportionalPolicy(Policy):
    def __init__(self):
        super().__init__('Proportional')

    def step(self, mab):
        times_of_pull = mab.get_number_of_pull_per_machine()
        return self.argmax(np.array([mab.prob_x[x] / times_of_pull[x] for x in range(mab.k)]))
