import numpy as np
from utils.evaluate import mean_roc_auc


def test_mean_roc_auc():
    y_true = np.random.randint(2, size=(200, 6))
    y_pred_proba = np.random.rand(6, 200, 2)
    score = mean_roc_auc(y_true, y_pred_proba)

    assert isinstance(score, np.float64)
    assert 0 <= score <= 1
