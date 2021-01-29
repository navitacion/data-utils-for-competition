import os
import numpy as np
import pandas as pd


def load_data(data_dir, sampling=1.0, seed=0, id_col=None, target_col='target'):
    """
    Load Raw Data
    Assumpt 2 Dataset 'train.csv' and 'test.csv'
    ---------------------------------------------
    Parameter
    data_dir: str
        directory of dataset
    sampling: float
        sampling train data
        for Debug
    seed: int
        Random Seed
    id_col: str
        column of Unique Id
        if id_col is None, 'id' columns add original dataset
    target_col: str
        column of target

    ---------------------------------------------
    Returns
    df: dataframe
        dataset (concatenated train, test)
    """
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    if id_col is None:
        train['id'] = np.arange(len(train))
        test['id'] = np.arange(len(test))
    test[target_col] = 0

    train['is_train'] = 1
    test['is_train'] = 0

    if sampling < 1:
        train = train.sample(frac=sampling, random_state=seed).reset_index(drop=True)

    df = pd.concat([train, test], axis=0, ignore_index=True)

    return df