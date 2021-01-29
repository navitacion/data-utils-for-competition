import os, gc
import numpy as np
import pandas as pd

class Trainer:
    def __init__(self, model, id_col, tar_col, g_col, features, cv, criterion, experiment):
        self.model = model
        self.id_col = id_col
        self.tar_col = tar_col
        self.g_col = g_col
        self.features = features
        self.cv = cv
        self.criterion = criterion
        self.experiment = experiment
        # Log params
        self.experiment.log_parameters(self.model.params)


    def _prepare_data(self, df, mode='fit'):
        """
        prepare dataset for training
        ---------------------------------------------
        Parameter
        df: dataframe
            preprocessed data
        mode: str
            If training, 'mode' set 'fit', else 'mode' set 'predict'

        ---------------------------------------------
        Returns
        X_train, y_train, X_test, train_id, test_id, features
        """
        assert 'is_train' in df.columns, 'Not contained is_train'
        # Split Train, Test Dataset
        train = df[df['is_train'] == 1].reset_index(drop=True)
        test = df[df['is_train'] == 0].reset_index(drop=True)
        drop_cols = ['is_train', self.g_col, self.id_col, self.tar_col]

        if self.features is None:
            self.features = [c for c in df.columns if c not in drop_cols]
        else:
            self.features = [c for c in self.features if c not in drop_cols]

        if mode == 'fit':
            # Train
            self.X_train = train[self.features].values
            self.y_train = train[self.tar_col].values
            self.y_train = self._transform_value(self.y_train, mode='forward')
            self.train_id = train[self.id_col].values
            self.group = train[self.g_col]

        if mode == 'predict':
            # Test
            self.X_test = test[self.features].values
            self.test_id = test[self.id_col].values


    def _transform_value(self, v, mode='forward'):
        """
        transform target values
        You need to calculate log, rewrite this function
        ---------------------------------------------
        Parameter
        v: int, float
            original values
        mode: str
            If calculating, 'mode' set 'forward'.
            Then reverse original values, 'mode' set 'backward'

        ---------------------------------------------
        Returns
        out: int, float
            Output value
        """
        if mode == 'forward':
            # out = np.log1p(v)
            out = v
        elif mode == 'backward':
            # v = np.where(v < 0, 0, v)
            # out = np.expm1(v)
            out = v
        else:
            out = v

        return out


    def _train_cv_group(self):
        """
        Train loop for Cross Validation
        """
        # init Model list
        self.models = []
        self.oof_pred = np.zeros(len(self.y_train))
        self.oof_y = np.zeros(len(self.y_train))

        # GroupKFold
        # ref: https://www.guruguru.science/competitions/13/discussions/cc7167cb-3627-448a-b9eb-7afcd29fd122/
        unique_year = self.group.unique()
        for i, (tr_group_idx, va_group_idx) in enumerate(self.cv.split(unique_year)):
            tr_groups, va_groups = unique_year[tr_group_idx], unique_year[va_group_idx]

            trn_idx = self.group.isin(tr_groups)
            val_idx = self.group.isin(va_groups)

            X_trn, y_trn = self.X_train[trn_idx], self.y_train[trn_idx]
            X_val, y_val = self.X_train[val_idx], self.y_train[val_idx]

            oof = self.model.train(X_trn, y_trn, X_val, y_val, feature_name=self.features)

            # Score
            oof = self._transform_value(oof, mode='backward')
            y_val = self._transform_value(y_val, mode='backward')
            score = self.criterion(y_val, oof)

            # Logging
            self.experiment.log_metric('Fold_score', score, step=i + 1)
            print(f'Fold {i + 1}  Score: {score:.3f}')
            self.oof_pred[val_idx] = oof
            self.oof_y[val_idx] = y_val
            self.models.append(self.model)


    def _train_cv(self):
        """
        Train loop for Cross Validation
        """
        # init Model list
        self.models = []
        self.oof_pred = np.zeros(len(self.y_train))
        self.oof_y = np.zeros(len(self.y_train))

        for i, (trn_idx, val_idx) in enumerate(self.cv.split(self.X_train, self.y_train)):
            X_trn, y_trn = self.X_train[trn_idx], self.y_train[trn_idx]
            X_val, y_val = self.X_train[val_idx], self.y_train[val_idx]

            oof = self.model.train(X_trn, y_trn, X_val, y_val, feature_name=self.features)

            # Score
            oof = self._transform_value(oof, mode='backward')
            y_val = self._transform_value(y_val, mode='backward')
            score = self.criterion(y_val, oof)

            # Logging
            self.experiment.log_metric('Fold_score', score, step=i + 1)
            print(f'Fold {i + 1}  Score: {score:.3f}')
            self.oof_pred[val_idx] = oof
            self.oof_y[val_idx] = y_val
            self.models.append(self.model)


    def _train_end(self):
        """
        End of Train loop per crossvalidation fold
        Logging and oof file
        """
        # Log params
        self.oof_score = self.criterion(self.oof_y, self.oof_pred)
        self.experiment.log_metric('Score', self.oof_score)
        print(f'All Score: {self.oof_score:.3f}')

        oof = pd.DataFrame({
            self.id_col: self.train_id,
            self.tar_col: self._transform_value(self.oof_pred, mode='backward')
        })

        oof = oof.sort_values(by=self.id_col)

        # Logging
        sub_name = f'oof_score_{self.oof_score:.4f}.csv'
        oof[[self.id_col, self.tar_col]].to_csv(os.path.join(sub_name), index=False)
        self.experiment.log_asset(file_data=sub_name, file_name=sub_name)
        os.remove(sub_name)


    def _predict_cv(self):
        """
        Predict loop for Cross Validation
        """
        assert len(self.models), 'You Must Train Something Model'
        self.preds = np.zeros(len(self.test_id))

        for m in self.models:
            pred = m.predict(self.X_test)
            self.preds += pred

        self.preds /= len(self.models)
        self.preds = self._transform_value(self.preds, mode='backward')


    def _predict_end(self):
        """
        End of Predict loop per crossvalidation fold
        Logging and submit file
        """
        sub = pd.DataFrame({
            self.id_col: self.test_id,
            self.tar_col: self.preds
        })

        sub = sub.sort_values(by=self.id_col)

        # Logging
        sub_name = f'sub_score_{self.oof_score:.4f}.csv'
        sub[[self.id_col, self.tar_col]].to_csv(os.path.join(sub_name), index=False)
        self.experiment.log_asset(file_data=sub_name, file_name=sub_name)
        os.remove(sub_name)

    def fit(self, df):
        self._prepare_data(df, mode='fit')
        if self.g_col is None:
            self._train_cv()
        else:
            self._train_cv_group()
        self._train_end()


    def predict(self, df):
        self._prepare_data(df, mode='predict')
        self._predict_cv()
        self._predict_end()


    def get_feature_importance(self):
        assert len(self.models) != 0, "You Must Train Model!!"
        feat_imp_df = pd.DataFrame(self.features, columns=['feature'])
        tar_col =[]

        for i, m in enumerate(self.models):
            # make DataFrame from Model Instance
            tmp_feat = pd.DataFrame({'feature': m.model.feature_name(), f'importance_{i}': m.get_feature_importance()})
            feat_imp_df = feat_imp_df.merge(tmp_feat, on='feature', how='left')
            # add Importance Name
            tar_col.append(f'importance_{i}')
        feat_imp_df.fillna(0, inplace=True)

        feat_imp_df['importance'] = feat_imp_df[tar_col].sum(axis=1)
        feat_imp_df['importance'] /= len(self.models)
        feat_imp_df = feat_imp_df[['feature', 'importance']]

        feat_imp_df.sort_values(by='importance', ascending=False, inplace=True)
        feat_imp_df.reset_index(drop=True, inplace=True)
        feat_imp_df.to_csv('feature_importance.csv', index=False)
        self.experiment.log_asset(file_data='feature_importance.csv', file_name='feature_importance.csv')
        os.remove('feature_importance.csv')
