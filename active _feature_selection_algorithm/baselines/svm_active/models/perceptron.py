"""
This module includes a class for interfacing scikit-learn's perceptron model.
"""
import sklearn.linear_model

from baselines.svm_active.base.interfaces import Model


class Perceptron(Model):

    """A interface for scikit-learn's perceptron model

    References
    ----------
    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html
    """

    def __init__(self, *args, **kwargs):
        self.model = sklearn.linear_model.Perceptron(*args, **kwargs)

    def train(self, dataset, *args, **kwargs):
        return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args),
                                **kwargs)
