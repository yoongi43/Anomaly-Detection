import os
import pandas as pd
import numpy as np

import seaborn as sns

sns.set(color_codes=True)
import matplotlib.pyplot as plt
from sklearn import preprocessing


def normalize(data, shuffle=True):
    scaler = preprocessing.MinMaxScaler()

    ndata = pd.DataFrame(scaler.fit_transform(data),
                         columns=data.columns,
                         index=data.index)

    if shuffle:
        ndata.sample(frac=1)  # random shuffle

    return ndata


def load(data_dir, train_proportion=0.8, showflag=False, normalizeflag=True, shuffleflag=True):
    """
    load dataset in the data_dir
    :param data_dir: Directory of dataset
    :param showflag: Show plot of dataset
    :param normalizeflag: Re-scale the data to be in the range [0, 1]
    :param shuffleflag: Shuffle the training dataset
    :return: train dataset, test dataset
    """
    data_dir = os.path.abspath(data_dir)
    parent_path = os.path.join(data_dir, '..')
    merged_data = pd.DataFrame()

    sample_dataset = pd.read_csv(os.path.join(data_dir, os.listdir(data_dir)[0]), sep='\t')
    num_col = sample_dataset.shape[1]
    # sample_mean_abs = np.array(sample_dataset.abs().mean())
    for filename in os.listdir(data_dir):
        dataset = pd.read_csv(os.path.join(data_dir, filename), sep='\t')
        dataset_mean_abs = np.array(dataset.abs().mean())
        dataset_mean_abs = pd.DataFrame(dataset_mean_abs.reshape(1, num_col))
        dataset_mean_abs.index = [filename]
        merged_data = merged_data.append(dataset_mean_abs)

    col_list = []
    for i in range(1, num_col + 1):
        col_list.append('Bearing {}'.format(i))
    merged_data.columns = col_list

    merged_data.index = pd.to_datetime(merged_data.index, format='%Y.%m.%d.%H.%M.%S')
    merged_data = merged_data.sort_index()

    merged_data.to_csv(os.path.join(parent_path, 'merged_dataset_BearingTest_{}.csv'.format(data_dir[-8])))

    print("Shape of merged_dataset: ", merged_data.shape)

    data_len = merged_data.shape[0]
    train_len = int(data_len * train_proportion)
    # test_len = int(data_len - train_len)
    dataset_train = merged_data[:train_len]
    dataset_test = merged_data[train_len:]

    if showflag:
        dataset_train.plot(figsize=(8, 4))
        plt.show()

    if normalizeflag:
        dataset_train = normalize(dataset_train, shuffle=shuffleflag)
        dataset_test = normalize(dataset_test, shuffle=False)

    return dataset_train, dataset_test


if __name__ == "__main__":
    # data_dir = './data/2nd_test'
    data_dir = './data/1st_test'
    load(data_dir, showflag=False)
