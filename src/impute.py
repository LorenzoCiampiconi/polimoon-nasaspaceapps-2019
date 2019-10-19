from sklearn.neighbors import KNeighborsRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import IterativeImputer

class NaiveImputer():
    def impute(self, column, df, Regressor=KNeighborsRegressor, **regr_kwargs):
        X = df.drop([column], axis=1)
        Y = df[column]

        pure_df = df.dropna()
        pure_x = pure_df.drop([column], axis=1)
        pure_y = pure_df[column]

        regr = Regressor(**regr_kwargs).fit(pure_x, pure_y)

        imputable = Y.isna().to_numpy() & X.notna().to_numpy().all(axis=1)
        if imputable.sum() == 0:
            return df, imputable
        x_t = X[imputable]
        y_t = regr.predict(x_t)
        # df.loc[imputable, column] = y_t
        return y_t, imputable, locals()


    def impute_all(self, df, Regressor=KNeighborsRegressor, **regr_kwargs):
        dffin = df.copy()
        for c in df.columns:
            y_t, imputable, _ = self.impute(c, df, Regressor=Regressor, **regr_kwargs)
            dffin.loc[imputable, c] = y_t
        return dffin


class ScikitImputer():
    def impute_all(self, df, regressor=None, **regr_kwargs):
        im = IterativeImputer(estimator=regressor)
        dffin = im.fit_transform(df)
        return dffin
