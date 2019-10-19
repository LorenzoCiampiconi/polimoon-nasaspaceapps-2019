import numpy as np

from src.impute import NaiveImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.special import softmax

class UCBImputer():
    def _scoreFeatures(self, df, y, cat):
        if cat:
            Regressor = RandomForestClassifier
        else:
            Regressor = RandomForestRegressor
        return Regressor(**self.regr_kwargs).fit(df, y).feature_importances_

    def impute_all(self, df, y=None, cat=False, feat_score=None, Regressor=None, ui_cost=1.5, **regr_kwargs):
        self.regr_kwargs = regr_kwargs
        if Regressor is None:
            if cat:
                Regressor = RandomForestClassifier
            else:
                Regressor = RandomForestRegressor
        if feat_score is None:
            feat_score = self._scoreFeatures(df, y, cat)
        feat_score = softmax(feat_score)

        dffin = df.copy().drop(df.index)
        pure_df = df.dropna()

        im = IterativeImputer(estimator=Regressor(**self.regr_kwargs))
        im = im.fit(pure_df)


        n = len(pure_df)
        stds = pure_df.vars()
        column_penalties = df.notna().to_numpy().sum(axis=0) + feat_score
        for r in df:
            r_id = r.notna().to_numpy()
            drop_penalty = r_id @ feat_score
            impute_penalty = 0
            for j, x in enumerate(r_id.flatten()):
                u = stds[j] * ui_cost / feat_score[j]
                impute_penalty += (~x) * np.exp(-2*n*u**2) * column_penalties[j]
            if drop_penalty <= impute_penalty:
                imputed = im.transform(r)
                dffin.append(r)
        return dffin







