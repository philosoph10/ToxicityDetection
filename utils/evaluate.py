import numpy as np
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import cross_val_score


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


def cross_validate(classifier, X, y, cv=3):
    """
    Perform Cross-Validation to test the model's performance
    :param classifier: the classification model
    :param X: texts for cross-validation
    :param y: labels for cross-validation
    :param cv: cross-validation strategy; default=3, meaning 3-fold cross-validation
    :return: the mean cross-validation score
    """
    custom_scorer = make_scorer(mean_roc_auc, greater_is_better=True, needs_proba=True)

    return np.mean(cross_val_score(classifier, X, y, cv=cv, scoring=custom_scorer))
