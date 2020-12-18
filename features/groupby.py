import itertools

from .base import BaseFeatureTransformer


class GroupbyTransformer(BaseFeatureTransformer):
    def __init__(self, group_cols, value_cols, aggs, conbination=1, use_cudf=False):
        super(GroupbyTransformer, self).__init__(use_cudf)
        self.group_cols = group_cols
        self.value_cols = value_cols
        self.aggs = aggs
        self.conbination = conbination


    def transform(self, df):
        # To categorical
        for g in self.group_cols:
            df[g] = df[g].astype('category')

        for v, agg in itertools.product(self.value_cols, self.aggs):
            for g in itertools.combinations(self.group_cols, self.conbination):
                g = list(g)
                g_text = '_'.join(g)
                col_name = f'fe_group_{g_text}_{v}_{agg}'
                group = df[g + [v]].groupby(g)[v].agg(agg).reset_index()
                group = group.rename(columns={v: col_name})
                df = df.merge(group, on=g, how='left')

        return df
