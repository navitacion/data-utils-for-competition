import category_encoders as ce

from .base import BaseFeatureTransformer


class CategoryEncoder(BaseFeatureTransformer):
    def __init__(self, cols=None):
        super(CategoryEncoder, self).__init__()
        self.cols = cols

    def transform(self, df):
        if self.cols is None or len(self.cols) == 0:
            self.cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        encoder = ce.OrdinalEncoder(cols=self.cols, handle_unknown='impute')
        df = encoder.fit_transform(df)

        for c in self.cols:
            df[c] = df[c].astype('category')

        return df


class FrequencyEncoder(BaseFeatureTransformer):
    def __init__(self, cols=None):
        super(FrequencyEncoder, self).__init__()
        self.cols = cols

    def transform(self, df):
        assert self.cols is not None or len(self.cols) == 0, "You must set 'col'"
        for c in self.cols:
            col_name = f'fe_freq_{c}'
            freq = df[df['is_train'] == 1][c].value_counts()
            df[col_name] = df[c].map(freq)

        return df


