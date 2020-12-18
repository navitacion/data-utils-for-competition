from abc import ABC, abstractmethod
import pickle
import cudf
from sklearn.base import BaseEstimator, TransformerMixin


class BaseFeatureTransformer(ABC, BaseEstimator, TransformerMixin):
    def __init__(self, use_cudf=False):
        self.use_cudf = use_cudf

    def __call__(self, df):
        df = cudf.from_pandas(df)
        df = self.transform(df)
        df = df.to_pandas(df)
        return df

    def fit(self, X):
        return self

    @abstractmethod
    def transform(self, df):
        raise NotImplementedError

    def get_categorical_features(self):
        return []

    def get_numerical_features(self):
        return []

