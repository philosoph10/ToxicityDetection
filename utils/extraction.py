import pandas as pd
import argparse
import os


def parse_arguments():
    """
    parse command line argument
    :return: the dictionary with the parsed arguments
    """
    parser = argparse.ArgumentParser(description="Parse command-line arguments")

    # Add command-line arguments
    parser.add_argument("--target-size", type=int, default=20000, help="Size of the target dataframe")
    parser.add_argument("--target-dir", type=str, default="./data", help="Directory to save the target dataframe")

    # Parse the arguments
    args_parsed = parser.parse_args()

    return args_parsed


def main(args, data_train):
    """
    Perform the extraction
    :param args: parsed cmd arguments
    :param data_train: all data
    """
    # define the label columns
    cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # extract all non-toxic rows
    zeros = data_train[data_train.apply(lambda row: len([col for col in cols if row[col] == 1]) == 0,
                                        axis=1)]

    # extract all toxic messages
    bad_messages = data_train[data_train.apply(lambda row: 1 in [row[col] for col in cols], axis=1)]

    # define the number of rows to extract
    EXTRACTION_TGT = args.target_size

    # calculate the number of non-toxic rows to extract
    num_non_toxic_rows_to_extract = EXTRACTION_TGT - bad_messages.shape[0]

    # sample non-toxic rows
    sample_good = zeros.sample(n=num_non_toxic_rows_to_extract, random_state=42)

    # define the extracted dataframe and save it
    extracted_df = pd.concat([sample_good, bad_messages])
    tgt_dir = args.target_dir
    if len(tgt_dir) < 4 or tgt_dir[-4:] != '.csv':
        os.makedirs(args.target_dir, exist_ok=True)
        tgt_dir = os.path.join(tgt_dir, 'Smart_Subset.csv')
    extracted_df.to_csv(tgt_dir)

    # print stats
    print(f"#total rows = {data_train.shape[0]}")
    print(f"#non-toxic rows = {zeros.shape[0]}")
    print(f"#toxic rows = {bad_messages.shape[0]}")
    print(f"#non-toxic rows used = {num_non_toxic_rows_to_extract}")


if __name__ == '__main__':
    # parse cmd arguments
    args_cmd = parse_arguments()

    # read 160k texts
    data = pd.read_csv('./data/train.csv')

    # extract essential data
    main(args_cmd, data)
