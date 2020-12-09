import numpy as np
from baselines.svm_active.base.dataset import Dataset
from baselines.svm_active.models import LogisticRegression
from baselines.svm_active.query_strategies import UncertaintySampling
from baselines.svm_active.labelers import IdealLabeler
from utility.utiliy_active import mi_score


def get_active_svm_index(X_train, y_train, total_labels):
    y_labes = np.array([None] * len(y_train))
    random_labels = []
    while len(np.unique(y_train[random_labels])) < 2:
        random_labels.append(np.random.choice(np.arange(len(y_labes))))

    y_labes[random_labels] = y_train[random_labels]
    fully_labeled_trn_ds = Dataset(X_train, y_train)
    lbr = IdealLabeler(fully_labeled_trn_ds)
    trn_ds = Dataset(X_train, y_labes)
    qs = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression())
    idx = run(trn_ds, lbr, qs, total_labels - len(random_labels))

    return idx + random_labels


def run(trn_ds, lbr, qs, quota):
    idx = []
    for _ in range(quota):
        ask_id = qs.make_query()
        idx.append(ask_id)
        lb = lbr.label(trn_ds.data[ask_id][0])
        trn_ds.update(ask_id, lb)

    return idx


def svm_active(X, y, k, prob_x, total_labels):
    svm_samples = get_active_svm_index(X, y, total_labels, )
    svm_score = np.zeros(total_labels)

    for i, rs in enumerate(svm_samples):
        samples_i = svm_samples[:i + 1]
        svm_score[i] = mi_score(X, y, k, prob_x, samples_i)

    return svm_score.tolist()
