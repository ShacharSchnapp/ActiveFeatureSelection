"""
This module uses real world dataset to test the active learning algorithms.
Since creating own artificial dataset for test is too time comsuming so we would
use real world data, fix the random seed, and compare the query sequence each
active learning algorithm produces.

The dataset yeast_train.svm is downloaded from
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html
"""
import os
import unittest

from numpy.testing import assert_array_equal
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MultiLabelBinarizer

from svm_active.base.dataset import Dataset
from baselines.svm_active.models import LogisticRegression, SVM
from baselines.svm_active.models import BinaryRelevance
from svm_active.query_strategies.multilabel import (
    AdaptiveActiveLearning,
    BinaryMinimization, 
    CostSensitiveReferencePairEncoding,
    MMC,
    MultilabelWithAuxiliaryLearner,
)
from ...tests.utils import run_qs
from svm_active.utils.multilabel import pairwise_f1_score


class MultilabelRealdataTestCase(unittest.TestCase):

    def setUp(self):
        dataset_filepath = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'datasets/yeast_train.svm')
        X, y = load_svmlight_file(dataset_filepath, multilabel=True)
        self.X = X.todense().tolist()
        self.y = MultiLabelBinarizer().fit_transform(y).tolist()
        self.quota = 10

    def test_mmc(self):
        trn_ds = Dataset(self.X,
                         self.y[:5] + [None] * (len(self.y) - 5))
        qs = MMC(trn_ds, random_state=1126)
        qseq = run_qs(trn_ds, qs, self.y, self.quota)
        assert_array_equal(qseq,
                np.array([117, 655, 1350, 909, 1003, 1116, 546, 1055, 165, 1441]))

    def test_multilabel_with_auxiliary_learner_hlr(self):
        trn_ds = Dataset(self.X,
                         self.y[:5] + [None] * (len(self.y) - 5))
        qs = MultilabelWithAuxiliaryLearner(trn_ds,
                major_learner=BinaryRelevance(
                        LogisticRegression(solver='liblinear',
                                           multi_class="ovr",
                                           random_state=1126)),
                auxiliary_learner=BinaryRelevance(SVM(gamma="auto")),
                criterion='hlr',
                random_state=1126)
        qseq = run_qs(trn_ds, qs, self.y, self.quota)
        assert_array_equal(qseq,
                np.array([701, 1403, 147, 897, 974, 1266, 870, 703, 292, 1146]))

    def test_multilabel_with_auxiliary_learner_shlr(self):
        trn_ds = Dataset(self.X,
                         self.y[:5] + [None] * (len(self.y) - 5))
        qs = MultilabelWithAuxiliaryLearner(trn_ds,
                major_learner=BinaryRelevance(LogisticRegression(solver='liblinear',
                                                                 multi_class="ovr")),
                auxiliary_learner=BinaryRelevance(SVM(gamma="auto")),
                criterion='shlr',
                b=1.,
                random_state=1126)
        qseq = run_qs(trn_ds, qs, self.y, self.quota)
        assert_array_equal(qseq,
                np.array([1258, 805, 459, 550, 783, 964, 736, 1004, 38, 750]))

    def test_multilabel_with_auxiliary_learner_mmr(self):
        trn_ds = Dataset(self.X,
                         self.y[:5] + [None] * (len(self.y) - 5))
        qs = MultilabelWithAuxiliaryLearner(trn_ds,
                major_learner=BinaryRelevance(LogisticRegression(solver='liblinear',
                                                                 multi_class="ovr")),
                auxiliary_learner=BinaryRelevance(SVM(gamma="auto")),
                criterion='mmr',
                random_state=1126)
        qseq = run_qs(trn_ds, qs, self.y, self.quota)
        assert_array_equal(qseq,
                np.array([1258, 1461, 231, 1198, 1498, 1374, 955, 1367, 265, 144]))

    def test_binary_minimization(self):
        trn_ds = Dataset(self.X, self.y[:5] + [None] * (len(self.y) - 5))
        qs = BinaryMinimization(trn_ds, LogisticRegression(solver='liblinear', multi_class="ovr"),
                                random_state=1126)
        qseq = run_qs(trn_ds, qs, self.y, self.quota)
        assert_array_equal(qseq,
                np.array([936, 924, 1211, 1286, 590, 429, 404, 962, 825, 30]))

    def test_adaptive_active_learning(self):
        trn_ds = Dataset(self.X, self.y[:5] + [None] * (len(self.y) - 5))
        qs = AdaptiveActiveLearning(trn_ds,
                base_clf=LogisticRegression(solver='liblinear', multi_class="ovr"), n_jobs=-1,
                                            random_state=1126)
        qseq = run_qs(trn_ds, qs, self.y, self.quota)
        assert_array_equal(qseq,
                np.array([594, 827, 1128, 419, 1223, 484, 96, 833, 37, 367]))


    def test_cost_sensitive_random_pair_encoding(self):
        trn_ds = Dataset(self.X, self.y[:5] + [None] * (len(self.y) - 5))
        model = BinaryRelevance(LogisticRegression(solver='liblinear',
                                                   multi_class="ovr"))
        base_model = LogisticRegression(
                solver='liblinear', multi_class="ovr", random_state=1126)
        qs = CostSensitiveReferencePairEncoding(
                trn_ds,
                scoring_fn=pairwise_f1_score,
                model=model,
                base_model=base_model,
                n_models=10,
                n_jobs=1,
                random_state=1126)
        qseq = run_qs(trn_ds, qs, self.y, self.quota)
        assert_array_equal(qseq,
                np.array([149, 434, 1126, 719, 983, 564, 816, 732, 101, 1242]))



if __name__ == '__main__':
    unittest.main()
