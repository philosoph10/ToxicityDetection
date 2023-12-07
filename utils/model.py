from sentence_transformers import SentenceTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression


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

    def predict(self, X):
        """
        Perform inference
        :param X: Feature vector
        :return: Predictions
        """
        # Embed the texts
        embeddings_X = self.model.encode(X)

        # Make predictions
        return self.classifier.predict(embeddings_X)

    def predict_proba(self, X):
        """
        Find probabilities of labels
        :param X: Feature vector
        :return: Probabilities for output labels
        """
        # Embed the texts
        embeddings_X = self.model.encode(X)

        # Make predictions
        return self.classifier.predict_proba(embeddings_X)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        :param deep: If True, will return the parameters for this estimator and contained subobjects
        :return: Parameter names mapped to their values
        """
        return {
            'model_name': self.model_name
        }
