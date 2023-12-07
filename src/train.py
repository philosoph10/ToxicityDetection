import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.model import ToxicityDetector
from utils.evaluate import mean_roc_auc


def parse_args():
    parser = argparse.ArgumentParser(description="Command-line arguments for your script")

    parser.add_argument("--data-path", default="./data/Smart_Subset.csv", help="Path to the dataset")
    parser.add_argument("--save-dir", required=True, help="Directory for saving the model")

    args = parser.parse_args()
    return args


def train(data_path, save_dir):
    """
    train the model
    :param data_path: path to the dataset, should contain at least 2500 words, excluding stopwords
    :param save_dir: the directory for saving the model, preferably empty
    """
    # read the data
    data = pd.read_csv(data_path)

    # prepare data for training
    classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    y = data[classes].to_numpy()
    texts = [text for text in data['comment_text']]

    # split the data into train and validation parts
    train_texts, test_texts, y_train, y_test = train_test_split(texts, y, test_size=0.2, random_state=42)

    # create the model instance
    detector = ToxicityDetector()

    # fit the model
    detector.fit(train_texts, y_train)

    # print stats
    print(f"Train mean roc-auc: {mean_roc_auc(y_train, detector.predict_proba(train_texts))}")
    print(f"Validation mean roc-auc: {mean_roc_auc(y_test, detector.predict_proba(test_texts))}")

    # save the model
    detector.save(save_dir)


if __name__ == '__main__':
    # parse cmd arguments
    args_parsed = parse_args()

    # train the model with the given arguments
    train(args_parsed.data_path, args_parsed.save_dir)
