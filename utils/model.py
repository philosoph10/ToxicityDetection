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
        :param model_name: model for embedding the text, default='intfloat/multilingual-e5-base'
        """
        self.model = SentenceTransformer(model_name)
        self.classifier = MultiOutputClassifier(LogisticRegression(class_weight='balanced', random_state=42))

    def fit(self, X, y):
        """
        Fit the model
        :param X: training data
        :param y: labels
        """
        # Embed the texts
        embeddings_X = self.model.encode(X)

        # Train the model
        self.classifier.fit(embeddings_X, y)

    def predict(self, X):
        """
        Perform inference
        :param X: Feature vector
        :return: predictions
        """
        # Embed the texts
        embeddings_X = self.model.encode(X)

        # Make predictions
        return self.classifier.predict(embeddings_X)

    def predict_proba(self, X):
        """
        Find probabilities of labels
        :param X: Feature vector
        :return: probabilities for output labels
        """
        # Embed the texts
        embeddings_X = self.model.encode(X)

        # Make predictions
        return self.classifier.predict_proba(embeddings_X)
