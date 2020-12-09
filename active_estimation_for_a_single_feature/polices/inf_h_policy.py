from polices.policy import Policy
import numpy as np


class INFHPolicy(Policy):
    def __init__(self, delta=0.05):
        super().__init__('MAX-H')
        self.delta = delta

    def step(self, mab):
        times_fo_pull = mab.get_number_of_pull_per_machine()
        score = (mab.get_sigma() + 3 * np.sqrt(np.log(1 / self.delta) / times_fo_pull)) / times_fo_pull
        return self.argmax(score)
