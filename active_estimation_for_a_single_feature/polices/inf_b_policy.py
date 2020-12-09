from polices.policy import Policy
import numpy as np


class INFBPolicy(Policy):
    def __init__(self, T, delta=0.05):
            super().__init__('MAX-B')
            self.delta = delta
            self.T = T

    def step(self, mab, c1=1, c2=np.e):
        times_fo_pull = mab.get_number_of_pull_per_machine()
        a1 = np.sqrt(2 * c1 * np.log(c2 / self.delta))
        a2 = (np.sqrt(c1 * self.delta * (1 + c2 + np.log(c2/self.delta)))) / ((1 - self.delta) * np.sqrt(2 * np.log(2 / self.delta)))
        a = a1 + a2*np.sqrt(self.T)
        sigma = np.sqrt(mab.get_sigma())
        score1 = sigma**2
        score2 = 4 * a * sigma * np.sqrt(np.log(2/self.delta) / times_fo_pull)
        score3 = 4 * a**2 * (np.log(2/self.delta) / times_fo_pull)
        score = (score1 + score2 + score3) / times_fo_pull
        return self.argmax(score)
