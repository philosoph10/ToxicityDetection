import numpy as np
from sklearn.metrics import roc_auc_score


def mean_roc_auc(y_true, y_pred_proba):
    """
    Calculate mean column-wise ROC AUC Score
    :param y_true: true labels
    :param y_pred_proba: predicted probabilities
    :return: mean column-wise ROC AUC Score
    """
    roc_auc_scores = []

    y_pred_proba = np.array(y_pred_proba)
    y_pred_proba = y_pred_proba[:, :, 1]
    y_pred_proba = np.swapaxes(y_pred_proba, 0, 1)

    for i in range(y_true.shape[1]):
        roc_auc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
        roc_auc_scores.append(roc_auc)
    return np.mean(roc_auc_scores)
