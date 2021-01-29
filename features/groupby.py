import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
from .base import BaseFeatureTransformer


class GroupbyTransformer(BaseFeatureTransformer):
    def __init__(self, group_cols, value_cols, aggs, conbination=1, use_cudf=False):
        super(GroupbyTransformer, self).__init__(use_cudf)
        # 重複を除外する
        self.group_cols = list(set(group_cols))
        self.value_cols = list(set(value_cols))
        self.aggs = aggs
        self.conbination = conbination


    def transform(self, df):
        # To categorical
        for g in self.group_cols:
            df[g] = df[g].astype('category')

        # 組み合わせ数を計算
        totals = int(len(self.value_cols) * len(self.aggs))

        # 一時保存用
        res = []
        cols = []

        for v, agg in tqdm(itertools.product(self.value_cols, self.aggs), total=totals):
            for g in itertools.combinations(self.group_cols, self.conbination):
                g = list(g)
                g_text = '_'.join(g)
                col_name = f'fe_group_{g_text}_{v}_{agg}'

                # Groupbyで集約
                group = df[g + [v]].groupby(g)[v].agg(agg).reset_index()
                group = group.rename(columns={v: col_name})

                # もとのデータセットにマージ
                tmp = df[['ID'] + g]
                tmp = tmp.merge(group, on=g, how='left')

                # 該当の値だけnumpyで取得する
                res.append(tmp[col_name].values)
                cols.append(col_name)

                del tmp, group

        # 列方向に結合
        res = np.stack(res, 1)
        res = pd.DataFrame(res, columns=cols)
        df = pd.concat([df, res], axis=1)

        return df
