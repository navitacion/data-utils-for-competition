
import pandas as pd

from utils import load_data
from utils import unpickle


from features import CategoryEncoder, FrequencyEncoder, GroupbyTransformer, LagTransformer, TextVectorizer

# pd.set_option('display.max_rows', None)
#
# data_dir = './input'
# df = load_data(data_dir, down_sample=1.0, seed=0, target_col='Global_Sales')
#
# cat_enc = LagTransformer(
#     value_cols=['Critic_Score', 'Critic_Count', 'User_Count'],
#     time_cols=['Year_of_Release'],
#     aggs=['mean'],
#     lags=[1],
# )
#
#
# text_vec = TextVectorizer(target_col='Name')
#
# df = cat_enc.transform(df)
# df = text_vec.transform(df)
#
# df.head(100).to_csv('ttt.csv')

df = unpickle('./input/data.pkl')

df.head(10).to_csv('uuu.csv')