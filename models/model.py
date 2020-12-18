import lightgbm as lgb
from catboost import Pool, CatBoostRegressor, CatBoostClassifier

from abc import ABCMeta, abstractmethod


# Basis -----------------------------------------------------------------------------------------------
class BaseModel(metaclass=ABCMeta):
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_test):
        raise NotImplementedError

    def get_feature_importance(self):
        pass


# LightGBM -----------------------------------------------------------------------------------------------
class LGBMModel(BaseModel):
    def __init__(self, params):
        super(LGBMModel, self).__init__(params)

    def train(self, X_train, y_train, X_val, y_val):
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        self.model = lgb.train(self.params,
                          train_data,
                          valid_sets=[valid_data, train_data],
                          valid_names=['eval', 'train'],
                          verbose_eval=1000,
                          )

        oof = self.model.predict(X_val, num_iteration=self.model.best_iteration)

        return oof


    def predict(self, X_test):
        pred = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        return pred

    def get_feature_importance(self):
        return self.model.feature_importance()


# CatBoost -----------------------------------------------------------------------------------------------
class CatBoostModel(BaseModel):
    def __init__(self, params):
        super(CatBoostModel, self).__init__(params)

    def train(self, X_train, y_train, X_val, y_val):
        train_data = Pool(X_train, label=y_train)
        valid_data = Pool(X_val, label=y_val)
        self.model = CatBoostRegressor(**self.params)
        self.model.fit(train_data,
                  eval_set=valid_data,
                  use_best_model=True)

        oof = self.model.predict(X_val)

        return oof

    def predict(self, X_test):
        pred = self.model.predict(X_test)
        return pred
