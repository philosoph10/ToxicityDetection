import numpy as np
# from sklearn.datasets import load_iris
# from sklearn.linear_model import LogisticRegression
# from sklearn.multioutput import MultiOutputClassifier

from utils.evaluate import mean_roc_auc  # , cross_validate
# from utils.model import ToxicityDetector


def test_mean_roc_auc():
    y_true = np.random.randint(2, size=(200, 6))
    y_pred_proba = np.random.rand(6, 200, 2)
    score = mean_roc_auc(y_true, y_pred_proba)

    assert isinstance(score, np.float64)
    assert 0 <= score <= 1


# def test_cross_validate():
#     X = ["Hello! My name is Yarema.", "How are you?", "This sentence is very-very toxic."]
#     y = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]]).reshape(3, 6)
#     classifier = ToxicityDetector()
#     # X = np.array([1, 2, 3, 4, 5, 6]).reshape(6, 1)
#     # y = np.array([[True, True], [True, False], [False, False], [False, True],
#     #               [True, True], [False, True]]).reshape(6, 2)
#     # X, y = load_iris(return_X_y=True)
#     # classifier = MultiOutputClassifier(LogisticRegression())
#     score = cross_validate(classifier, X, y)
#
#     assert isinstance(score, np.float64)
#     assert 0 <= score <= 1
