import itertools

from .base import BaseFeatureTransformer

class LagTransformer(BaseFeatureTransformer):
    def __init__(self, value_cols, time_cols, aggs, lags):
        super(LagTransformer, self).__init__()
        self.value_cols = value_cols
        self.time_cols = time_cols
        self.aggs = aggs
        self.lags = lags


    def transform(self, df):
        for v, lag, agg, time_col in itertools.product(self.value_cols, self.lags, self.aggs, self.time_cols):
            col_name = f'fe_lag_{lag}_{v}_{agg}'
            group = df.groupby(time_col)[v].agg(agg).shift(lag).reset_index()
            group = group.rename(columns={v: col_name})
            df = df.merge(group, on=time_col, how='left')

        return df


class DiffTransformer(BaseFeatureTransformer):
    def __init__(self, value_cols, time_cols, aggs, lags):
        super(DiffTransformer, self).__init__()
        self.value_cols = value_cols
        self.time_cols = time_cols
        self.aggs = aggs
        self.lags = lags


    def transform(self, df):
        for v, lag, agg, time_col in itertools.product(self.value_cols, self.lags, self.aggs, self.time_cols):
            col_name = f'fe_diff_{lag}_{v}_{agg}'
            group = df.groupby(time_col)[v].agg(agg).diff(lag).reset_index()
            group = group.rename(columns={v: col_name})
            df = df.merge(group, on=time_col, how='left')

        return df


class LagRollingTransformer(BaseFeatureTransformer):
    def __init__(self, value_cols, time_cols, aggs, lags, windows):
        super(LagRollingTransformer, self).__init__()
        self.value_cols = value_cols
        self.time_cols = time_cols
        self.aggs = aggs
        self.lags = lags
        self.windows = windows


    def transform(self, df):
        for v, lag, agg, time_col, w in itertools.product(self.value_cols, self.lags, self.aggs, self.time_cols, self.windows):
            col_name = f'fe_lag_{lag}_roll_{w}_{v}_{agg}'
            group = df.groupby(time_col)[v].agg(agg).shift(lag).rolling(window=w).reset_index()
            group = group.rename(columns={v: col_name})
            df = df.merge(group, on=time_col, how='left')

        return df