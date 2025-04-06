import lightgbm as lgb


class LightGBMModel:

    def __init__(self, params=None):
        # Use provided params or set defaults
        if params is None:
            params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'n_estimators': 100,
                'random_state': 42,

            }
        self.model = lgb.LGBMRegressor(**params)

    def train(self, x, y):
        # Drop NaNs and Infs
        y = y.dropna()
        x = x.loc[y.index]
        
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

