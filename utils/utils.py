import os
import re
import random
import pickle
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def reduce_mem_usage(df, verbose=False):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    with np.errstate(invalid='ignore'):
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    # elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    #     df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
            start_mem - end_mem) / start_mem))
    return df


def prep_text(df):
    corpus = df['Name'].tolist()
    corpus = [re.sub('[:()!.-]', '', str(text)) for text in corpus]
    corpus = [text.lower() for text in corpus]

    remove_words = [' of ', ' the ', ' in ', ' a ', ' an ', ' vol', "'s"]
    for w in remove_words:
        corpus = [text.replace(w, ' ') for text in corpus]

    corpus = [text.replace(' iii', ' 3') for text in corpus]
    corpus = [text.replace(' ii', ' 2') for text in corpus]
    corpus = [text.replace(' iv', ' 4') for text in corpus]
    corpus = [text.replace(' v ', ' 5 ') for text in corpus]
    corpus = [text.replace(' i ', ' 1 ') for text in corpus]

    corpus = [text.replace('   ', ' ') for text in corpus]
    corpus = [text.replace('  ', ' ') for text in corpus]

    df['Name'] = corpus

    return df


def to_pickle(filename, obj):
    with open(filename, mode='wb') as f:
        pickle.dump(obj, f)

def unpickle(filename):
    with open(filename, mode='rb') as fo:
        p = pickle.load(fo)
    return p