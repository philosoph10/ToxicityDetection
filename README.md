# Toxicity Detection Solution for Kaggle Challenge

![Project Logo](https://gopostr.s3.amazonaws.com/favicon_url/86gvpmIik9SpYAtIGFIgB0NUogeSA90n3Se6Hi4H.png)

## Overview

This GitHub repository provides a solution to the Kaggle Toxic Comment Classification Challenge, achieving a mean ROC-AUC score of 0.98 over six toxicity classes on the public dataset. The solution is implemented in Python 3.9 and Python 3.10.

## Models

The toxicity detection solution is an ensemble of two models:

1. **Transformer Model with Logistic Regression:**
   - Utilizes the SentenceTransformer library with the 'intfloat/multilingual-e5-base' model.
   - A logistic regression classifier is applied over the features extracted by the transformer.

2. **Pipeline Model:**
   - Consists of three elements: TF-IDF vectorizer, Select K Best, and Logistic Regression.
   - Provides an alternative approach to the problem using a different set of techniques.

## Installation

Follow these steps to set up the project:

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Extract a subset of the data:
Run the extraction script with optional command line arguments:

- `--target-size`: Size of the extracted dataset (default: $20,000$)
- `--target-dir`: Target directory for the extracted data (default: `./data`)

## Project Structure

The project is organized into several folders:

- **EDA (Exploratory Data Analysis):**
  - Contains a Jupyter notebook with in-depth analysis of the dataset.

- **Utils:**
  - Includes utility functions, model definitions, and data preparation scripts.
  - Recommends extracting 20k samples from the entire dataset, focusing on samples exhibiting toxicity.

- **Config:**
  - Holds a pretrained model ready for inference.
  - Load the model like this:
    ```python
    from toxicity_detector import ToxicityDetector
    detector = ToxicityDetector()
    detector.load('./config/best_model')
    ```

- **Src (Source Code):**
  - Contains two main scripts:
    - **Training Script:**
      - Accepts two command line arguments:
        - `--data-path`: Path to the training dataset (default: `./data/Smart_Subset.csv`)
        - `--save-dir`: Path to the directory for saving the model (required)
      - Sample run:
        ```bash
        python src/train.py --data-path ./data/Smart_Subset.csv --save-dir ./config/my_model
        ```
      - Trains and saves the model.

    - **Inference Script:**
      - Accepts one command line argument:
        - `--text`: String upon which to apply the model (required)
      - Sample run:
        ```bash
        python src/inference.py --text "You are a fool!"
        ```
      - Prints the verdict for each of the toxicity classes to the console.
