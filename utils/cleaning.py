import re
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')
stopwords = set(stopwords.words('english'))


def remove_special_symbols(texts):
    """
    remove special symbols - everything but letters and whitespaces
    :param texts: a list of strings
    :return: a list of filtered texts
    """
    special_symbol_pattern = r'[^A-Za-z\s]'  # anything that is not a letter or whitespace

    filtered_texts = []

    for text in texts:
        filtered_text = re.sub(special_symbol_pattern, '', text)
        filtered_text = re.sub(' +', ' ', filtered_text)
        filtered_texts.append(filtered_text.strip())

    return filtered_texts


def remove_stopwords(texts):
    """
    removes english stopwords from a list of strings
    turns texts to lowercase before applying the function to correctly detect the stopwords
    :param texts: a list of strings
    :return: texts without stopwords and turned to lowercase
    """
    cleaned_texts = []
    for text in texts:
        tokens = text.lower().split(' ')
        cleaned_tokens = [token for token in tokens if token not in stopwords]
        cleaned_texts.append(" ".join(cleaned_tokens))
    return cleaned_texts
