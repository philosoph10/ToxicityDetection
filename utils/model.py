from sentence_transformers import SentenceTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import joblib
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

    def save(self, save_dir):
        """
        Save the ToxicityDetector model to the specified directory.
        :param save_dir: Directory to save the model
        """
        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Save model name to a text file
        model_name_path = os.path.join(save_dir, 'model_name.txt')
        with open(model_name_path, 'w') as f:
            f.write(self.model_name)

        # Save sklearn pipeline
        sklearn_pipeline_path = os.path.join(save_dir, 'sklearn_pipeline.joblib')
        joblib.dump(self.ml_classifier, sklearn_pipeline_path)

        # Save MultiOutputClassifier
        multioutput_classifier_path = os.path.join(save_dir, 'multioutput_classifier.joblib')
        joblib.dump(self.classifier, multioutput_classifier_path)

    def load(self, load_dir):
        """
        Load the ToxicityDetector model from the specified directory.
        :param load_dir: Directory containing the saved model components
        """
        # Load model name from the text file
        model_name_path = os.path.join(load_dir, 'model_name.txt')
        with open(model_name_path, 'r') as f:
            self.model_name = f.read().strip()

        # Re-instantiate Sentence Transformer model based on the loaded model name
        self.model = SentenceTransformer(self.model_name)

        # Load sklearn pipeline
        sklearn_pipeline_path = os.path.join(load_dir, 'sklearn_pipeline.joblib')
        self.ml_classifier = joblib.load(sklearn_pipeline_path)

        # Load MultiOutputClassifier
        multioutput_classifier_path = os.path.join(load_dir, 'multioutput_classifier.joblib')
        self.classifier = joblib.load(multioutput_classifier_path)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        :param deep: If True, will return the parameters for this estimator and contained subobjects
        :return: Parameter names mapped to their values
        """
        return {
            'model_name': self.model_name
        }
