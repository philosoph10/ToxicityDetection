import argparse
from utils.model import ToxicityDetector


def main():
    parser = argparse.ArgumentParser(description='Toxicity Inference Script')
    parser.add_argument('--text', type=str, required=True, help='Input text for toxicity prediction')
    args = parser.parse_args()

    txt = args.text
    classifier = ToxicityDetector()
    classifier.load('./config/best_model')

    result = classifier.predict([txt])[0]
    print(f"toxic: {bool(result[0])}")
    print(f"severe toxic: {bool(result[1])}")
    print(f"obscene: {bool(result[2])}")
    print(f"threat: {bool(result[3])}")
    print(f"insult: {bool(result[4])}")
    print(f"identity hate: {bool(result[5])}")


if __name__ == "__main__":
    main()
