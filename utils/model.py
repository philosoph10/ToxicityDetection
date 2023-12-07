from sentence_transformers import SentenceTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from utils.cleaning import remove_special_symbols, remove_stopwords


class ToxicityDetector:
    """
    Model for toxicity detection
    Logistic regression on top of text embeddings
    """
    def __init__(self, model_name='intfloat/multilingual-e5-base'):
        """
        Initialize the model
        :param model_name: Model for embedding the text, default='intfloat/multilingual-e5-base'
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.classifier = MultiOutputClassifier(LogisticRegression(class_weight='balanced', random_state=42))
        self.ml_classifier = ToxicityDetector.get_ml_model()

    def fit(self, X, y):
        """
        Fit the model
        :param X: Training data
        :param y: Labels
        """
        # Embed the texts
        embeddings_X = self.model.encode(X)

        # Train the model
        self.classifier.fit(embeddings_X, y)

        # preprocess input for the 2nd model
        X_processed = ToxicityDetector.preprocess_text(X)

        self.ml_classifier.fit(X_processed, y)

    # noinspection SpellCheckingInspection
    def predict(self, X):
        """
        Perform inference
        :param X: Feature vector
        :return: Predictions
        """
        # Embed the texts
        embeddings_X = self.model.encode(X)

        probs = np.array(self.predict_proba(X))

        preds = np.zeros((X.shape[0], 6))

        for j in range(6):
            for i in range(X.shape[0]):
                preds[i][j] = 1 if probs[j][i][0] < probs[j][i][1] else 0

        # Make predictions
        return preds

    def predict_proba(self, X):
        """
        Find probabilities of labels
        :param X: Feature vector
        :return: Probabilities for output labels
        """
        # Embed the texts
        embeddings_X = self.model.encode(X)

        # preprocess the texts
        X_processed = ToxicityDetector.preprocess_text(X)

        # get the predicted probabilities
        probs_dl = np.array(self.classifier.predict_proba(embeddings_X))
        probs_ml = np.array(self.ml_classifier.predict_proba(X_processed))

        probs_av = (probs_dl + probs_ml) / 2.

        # Make predictions
        return [prob_sample for prob_sample in probs_av]

    @staticmethod
    def preprocess_text(texts):
        """
        preprocess input texts
        :param texts: a list of texts
        :return: a list of processed texts
        """
        return remove_stopwords(remove_special_symbols(texts))

    @staticmethod
    def get_ml_model():
        """
        a model utilizing pure ML methods, used for ensemble
        :return: the model
        """
        return Pipeline([
            ('vectorizer', TfidfVectorizer(sublinear_tf=True, max_df=0.5)),
            ('select', SelectKBest(chi2, k=2500)),
            ('moc', MultiOutputClassifier(estimator=LogisticRegression(class_weight='balanced')))
        ])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        :param deep: If True, will return the parameters for this estimator and contained subobjects
        :return: Parameter names mapped to their values
        """
        return {
            'model_name': self.model_name
        }
