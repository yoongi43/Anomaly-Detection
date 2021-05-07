import pandas as pd
import os


def read_dataframe(path_examples, data_idx):

    path_examples = os.path.abspath(path_examples)
    files_examples = os.listdir(path_examples)
    df = pd.read_csv(os.path.join(path_examples, files_examples[data_idx]),
                     parse_dates=['ds'],
                     dayfirst=False,
                     infer_datetime_format=True
                     )
    print("Load Dataframe:", files_examples[data_idx])

    return df
