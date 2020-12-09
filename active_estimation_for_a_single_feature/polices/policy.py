import numpy as np
from abc import ABC, abstractmethod
from mab_unit.mab import MAB
from utility.utiliy_active import mean_confidence_interval


class Policy(ABC):
    def __init__(self, name):
        self.name = name
        self.score_beta = []
        self.entropy_score = []
        self.avg_pull = []
        self.score = []

    @abstractmethod
    def step(self, mab: MAB):
        pass

    def get_p_hat(self, mab):
        return mab.get_p_hat()

    def run_places(self, T, mab):
        entropy_score = []
        for t in range(T):
            entropy_score.append(mab.entropy_lost(self.get_p_hat(mab)))
            x = self.step(mab)
            mab.pull_machine(x)

        self.entropy_score += [entropy_score]

    def get_avg(self):
        return mean_confidence_interval(np.array(self.entropy_score))

    @staticmethod
    def argmax(arr):
        return np.random.choice(np.flatnonzero(arr == arr.max()))

    @staticmethod
    def argmin(arr):
        return np.random.choice(np.flatnonzero(arr == arr.min()))