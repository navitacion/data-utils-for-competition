import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation, PCA, NMF


from .base import BaseFeatureTransformer



def get_vectorizer(name, ngram_range):
    vectorizer_dict = {
        'tfidf': TfidfVectorizer(ngram_range=ngram_range),
        'count': CountVectorizer(ngram_range=ngram_range)
    }

    return vectorizer_dict[name]


def get_transformer(name, n_components, seed):
    transformer_dict = {
        'pca': PCA(n_components=n_components, random_state=seed),
        'nmf': NMF(n_components=n_components, random_state=seed),
        'svd': TruncatedSVD(n_components=n_components, random_state=seed),
        'lda': LatentDirichletAllocation(n_components=n_components, random_state=seed)
    }

    return transformer_dict[name]


class TextVectorizer(BaseFeatureTransformer):
    def __init__(self, target_col, vectorizer='tfidf', transformer='svd', ngram_range=(1, 1), n_components=100, seed=0):
        super(TextVectorizer, self).__init__()

        self.vectorizer = get_vectorizer(name=vectorizer, ngram_range=ngram_range)
        self.transformer = get_transformer(name=transformer, n_components=n_components, seed=seed)
        self.target_col = target_col
        self.n_components = n_components
        self.name = f"fe_textvec_{vectorizer}-{transformer}_{ngram_range}_{self.target_col}"

    def transform(self, df):
        corpus = df[self.target_col].fillna('NaN').tolist()

        corpus = self.vectorizer.fit_transform(corpus)
        feature = self.transformer.fit_transform(corpus)
        out = pd.DataFrame(feature, columns=[f"{self.name}_{i}" for i in range(self.n_components)])
        df = pd.concat([df, out], axis=1)

        return df


class PivotCountEncoder(BaseFeatureTransformer):
    def __init__(self, index_col, count_cols, value_col, transformer='svd', n_components_rate=0.1, seed=0, use_cudf=False):
        super(PivotCountEncoder, self).__init__(use_cudf)
        self.index_col = index_col
        self.count_cols = count_cols
        self.value_col = value_col
        self.transformer = transformer
        self.n_components_rate = n_components_rate
        self.seed = seed
        self.name = f"fe_pivotcount_{self.index_col}_{transformer}"

    def transform(self, df):
        uniques = 0
        for c in self.count_cols:
            uniques += df[c].nunique()
        n_components = int(uniques * self.n_components_rate)

        for i, c in enumerate(self.count_cols):
            plat_pivot = df.pivot_table(index=self.index_col,
                                        columns=c,
                                        values=self.value_col,
                                        aggfunc='count', fill_value=0).reset_index().add_prefix(f'Count_{c}_')
            plat_pivot = plat_pivot.rename(columns={f'Count_{c}_{self.index_col}': self.index_col})
            if i == 0:
                temp = plat_pivot.copy()
            else:
                temp = pd.merge(temp, plat_pivot, on=[self.index_col], how='left')

        temp.fillna(0, inplace=True)

        transformer = get_transformer(name=self.transformer, n_components=n_components, seed=self.seed)

        tar_cols = [c for c in temp.columns if c.startswith('Count_')]
        out = transformer.fit_transform(temp[tar_cols].values)

        out = pd.DataFrame(out, columns=[f"{self.name}_{i}" for i in range(n_components)])
        out[self.index_col] = temp[self.index_col]
        del temp

        df = pd.merge(df, out, on=[self.index_col], how='left')

        return df