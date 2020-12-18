import os
import numpy as np
from comet_ml import Experiment
import hydra
from omegaconf import DictConfig

import pandas as pd

from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_log_error

from utils.utils import unpickle, to_pickle, seed_everything, prep_text
from utils.data import load_data
from features import CategoryEncoder, FrequencyEncoder, GroupbyTransformer, TextVectorizer, PivotCountEncoder

from models.trainer import Trainer
from models.model import LGBMModel, CatBoostModel

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', None)


def RMLSE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_pred = np.where(y_pred < 0, 0, y_pred)
    score = np.sqrt(mean_squared_log_error(y_true, y_pred))
    return score


def preprocessing(df, cfg):
    # 変なデータは削除
    df = df[df['Name'] != 'Strongest Tokyo University Shogi DS']
    df = df.reset_index(drop=True)
    df.loc[df['Name'] == 'Imagine: Makeup Artist', 'Year_of_Release'] = np.nan

    # Nameを前処理
    df['Name'] = df['Name'].fillna('unknown')
    df = prep_text(df)

    # TODO 発売から経過年数
    df['fe_diff_from_now'] = 2020 - df['Year_of_Release']

    # TODO tbdかどうか
    df['fe_is_tbd'] = df['User_Score'].apply(lambda x: 1 if x == 'tbd' else 0)

    # TODO ハードが携帯端末かどうか
    mobile = ['DS', 'PSP', 'GBA', '3DS', 'PSV', 'GB']
    df['fe_is_mobile'] = df['Platform'].apply(lambda x: 1 if x in mobile else 0)

    # TODO 年代をカテゴリ化
    dev_year = [0, 1992, 1999, 2004, 2009, 2021]
    df['fe_cat_year'] = pd.cut(df['Year_of_Release'], dev_year, labels=False)
    df.loc[(df['Year_of_Release'].isnull()), 'fe_cat_year'] = np.nan

    # TODO ユーザースコアのtbdを-1とし、数値に変換する
    df['User_Score'] = df['User_Score'].replace('tbd', -1)
    df['User_Score'] = df['User_Score'].astype(float)

    # TODO スコアと数の掛け合わせた変数
    df['fe_product_critic_score_count'] = df['Critic_Score'] * df['Critic_Count']
    df['fe_product_user_score_count'] = df['User_Score'] * df['User_Count']

    # TODO ある単語が入っているかどうかの変数
    target_name = ['soccer', 'dragon', 'wars', 'ds', 'battle', 'disney', 'lego',
                   'collection', 'party', 'ultimate', 'edition', 'baseball', 'fantasy',
                   'gundam', 'legend', 'mario', 'ninja', 'monster', 'sonic', 'samurai',
                   'tennis', 'batman', 'harry', 'yugioh', 'assassin']
    for t in target_name:
        df[f'fe_is_{t}_in_name'] = df['Name'].apply(lambda x: 1 if t in x.lower() else 0)

    # TODO ソフト名をテキストマイニング
    print('text')
    text_vectorizer = TextVectorizer(target_col='Name',
                                     vectorizer='tfidf',
                                     transformer='svd',
                                     ngram_range=(1, 3),
                                     n_components=cfg.data.vec_n_components)
    df = text_vectorizer.transform(df)


    # TODO Publisher単位の分布個数に応じた変数
    count_cols = ['Year_of_Release', 'Platform', 'Genre', 'Developer', 'Rating']
    enc = PivotCountEncoder(index_col='Publisher', count_cols=count_cols, value_col='Name',
                            transformer='svd', n_components_rate=0.2,
                            seed=cfg.data.seed, use_cudf=True)
    df = enc.transform(df)

    count_cols = ['Year_of_Release', 'Platform', 'Genre', 'Publisher', 'Rating']
    enc = PivotCountEncoder(index_col='Developer', count_cols=count_cols, value_col='Name',
                            transformer='svd', n_components_rate=0.2,
                            seed=cfg.data.seed, use_cudf=True)
    df = enc.transform(df)


    # TODO KMeans
    tar_cols = [c for c in df.columns if c.startswith('fe_pivotcount_Publisher_')]
    kmeans = KMeans(n_clusters=10)
    df['fe_cat_Publisher_count_kmeans'] = kmeans.fit_predict(df[tar_cols].fillna(0))

    # TODO Developer単位の分布個数結果をKMeans
    tar_cols = [c for c in df.columns if c.startswith('fe_pivotcount_Developer_')]
    kmeans = KMeans(n_clusters=10)
    df['fe_cat_Developer_count_kmeans'] = kmeans.fit_predict(df[tar_cols].fillna(0))

    # TODO テキストマイニング結果をKMeans
    tar_cols = [c for c in df.columns if c.startswith('fe_textvec')]
    kmeans = KMeans(n_clusters=10)
    df['fe_cat_kmeans'] = kmeans.fit_predict(df[tar_cols])


    # TODO Groupbyでいろいろ集約
    print('groupby')
    group_cols = ['Platform', 'Genre', 'Developer', 'Rating', 'fe_cat_year', 'fe_cat_kmeans']
    value_cols = ['Critic_Score', 'Critic_Count', 'User_Score', 'User_Count',
                  'fe_product_critic_score_count', 'fe_product_user_score_count']
    aggs = ['mean', 'sum', 'std', 'max', 'min', 'nunique']

    for conbi in [1, 2]:
        transformer = GroupbyTransformer(group_cols, value_cols, aggs, conbination=conbi, use_cudf=True)
        df = transformer.transform(df)


    # TODO Frequency Encoding
    # trainとtestで分布が全く違うPublisherを入れるとスコア悪くなる！
    group_cols = ['Platform', 'Genre', 'Developer', 'Rating', 'Name']
    freq_enc = FrequencyEncoder(cols=group_cols)
    df = freq_enc.transform(df)


    # ---------- 下記はfeature_enginneringを一通りやったあとの処理 -------------------------------------------
    not_use_col = ['id', 'Global_Sales', 'is_train']

    # カテゴリ変数に
    cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if c not in not_use_col]
    cols += [c for c in df.columns if c.startswith('fe_is_')]
    cols += [c for c in df.columns if c.startswith('fe_cat_')]
    # 重複を削除
    cols = list(set(cols))
    cat_enc = CategoryEncoder(cols=cols)
    df = cat_enc.transform(df)

    return df


@hydra.main('config.yml')
def main(cfg: DictConfig):
    print('atmaCup #8 Model Training')
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)
    data_dir = './input'

    seed_everything(cfg.data.seed)

    experiment = Experiment(api_key=cfg.exp.api_key,
                            project_name=cfg.exp.project_name)

    experiment.log_parameters(dict(cfg.data))

    # Load Data  ####################################################################################
    if cfg.exp.use_pickle:
        # pickleから読み込み
        df = unpickle('./input/data.pkl')

    else:
        df = load_data(data_dir, down_sample=0.1, seed=cfg.data.seed)
        # Preprocessing
        print('Preprocessing')
        df = preprocessing(df, cfg)

        # pickle形式で保存
        to_pickle('./input/data.pkl', df)
        try:
            experiment.log_asset(file_data='./input/data.pkl', file_name='data.pkl')
        except:
            pass


    # Config  ####################################################################################
    del_tar_col = [
        'Name', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Publisher'
    ]
    features = [c for c in df.columns if c not in del_tar_col]
    id_col = 'id'
    tar_col = 'Global_Sales'
    criterion = RMLSE
    cv = KFold(n_splits=cfg.data.n_splits, shuffle=True, random_state=cfg.data.seed)

    # Model  ####################################################################################
    model = None
    if cfg.exp.model == 'lgb':
        model = LGBMModel(dict(cfg.lgb))
    elif cfg.exp.model == 'cat':
        model = CatBoostModel(dict(cfg.cat))

    # Train & Predict  ##############################################################################
    trainer = Trainer(model, id_col, tar_col, features, cv, criterion, experiment)
    trainer.fit(df)
    trainer.predict(df)
    trainer.get_feature_importance()


if __name__ == '__main__':
    main()
