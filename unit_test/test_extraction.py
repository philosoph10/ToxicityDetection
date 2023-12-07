import os.path
import pandas as pd
from utils.extraction import main


def test_main():
    class NamespaceObject:
        """
        class that simulates argparse output
        """
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    # use the default params
    args = NamespaceObject(target_size=20000, target_dir='../data')

    main(args)

    # check that the dataframe was created
    data_path = './data/Smart_Subset.csv'
    assert os.path.exists(data_path)

    extracted_data = pd.read_csv(data_path)

    # check that there are 20k rows
    assert extracted_data.shape[0] == 20000

    # assert that there are 3775 rows without any toxicity
    cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    assert extracted_data[cols].sum(axis=1).eq(0).sum() == 3775
