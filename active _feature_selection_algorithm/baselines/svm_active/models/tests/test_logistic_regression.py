""" Test Logistic Regression Model

Since Logistic Regression model is from scikit-learn, we test it by checking if
it has the same result as the model in scikit-learn on iris dataset.
"""
import unittest

from numpy.testing import assert_array_equal
from sklearn import datasets
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split
import sklearn.linear_model

from svm_active.base.dataset import Dataset
from baselines.svm_active.models import LogisticRegression


class LogisticRegressionIrisTestCase(unittest.TestCase):

    def setUp(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.3, random_state=1126)

    def test_logistic_regression(self):
        clf = sklearn.linear_model.LogisticRegression(
            solver='liblinear', multi_class="ovr")
        clf.fit(self.X_train, self.y_train)
        lr = LogisticRegression(solver='liblinear', multi_class="ovr")
        lr.train(Dataset(self.X_train, self.y_train))

        assert_array_equal(
            clf.predict(self.X_train), lr.predict(self.X_train))
        assert_array_equal(
            clf.predict(self.X_test), lr.predict(self.X_test))
        self.assertEqual(
            clf.score(self.X_train, self.y_train),
            lr.score(Dataset(self.X_train, self.y_train)))
        self.assertEqual(
            clf.score(self.X_test, self.y_test),
            lr.score(Dataset(self.X_test, self.y_test)))


if __name__ == '__main__':
    unittest.main()
