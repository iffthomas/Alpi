from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from statsmodels.tsa.arima.model import ARIMA

import xgboost as xgb

import pandas as pd
import numpy as np


class SimpleModel:
    """
    This is a simple example of a model structure

    """

    def __init__(self):
        self.linear_regression = LinearRegression()

    def train(self, x, y):
        self.linear_regression.fit(x, y)

    def predict(self, x):
        return self.linear_regression.predict(x)
    

class SVRModel:
    """
    This is a simple example of a model structure

    """

    def __init__(self):
        self.svr = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1.0, epsilon=0.1))

    def train(self, x, y):
        self.svr.fit(x, y)

    def predict(self, x):
        return self.svr.predict(x)

class RandomForestModel:
    def __init__(self, n_estimators=500, random_state=42):
        # Initialize the Random Forest Regressor
        self.random_forest = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    def train(self, x, y):
        # Fit the model to the training data
        self.random_forest.fit(x, y)

    def predict(self, x):
        # Predict using the trained model
        return self.random_forest.predict(x)

class ARIMAModel:
    """This is an ARIMA Model that is used for predicting a specific timeframe"""
    
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
        self.results = None

    def train(self, y):
        # Note: ARIMA models are usually univariate. 
        # y is expected to be a one-dimensional series (or array).
        if y.index.inferred_freq is None:
            y = y.asfreq('H')  # or any other frequency that fits your data
            y = y.fillna(y.mean())
        self.model = ARIMA(y, order=self.order)
        self.results = self.model.fit()

    def predict(self, start, end):
        # Produces forecasts from start to end
        return self.results.predict(start=start, end=end)


from statsmodels.tsa.statespace.sarimax import SARIMAX


class MultiTimeSeriesForecaster:
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24*7*4)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.results = None

    def train(self, y, X):
        # y: target series (Pandas Series)
        # X: exogenous variables (Pandas DataFrame with same index)
        y = y.dropna()
        X = X.loc[y.index]  # Align X with y
    
        y = y.fillna(y.mean())
        X = X.fillna(X.mean())

    

        self.model = SARIMAX(endog=y, exog=X, order=self.order, seasonal_order=self.seasonal_order)
        self.results = self.model.fit(disp=False)

    def predict(self, start, end, exog):
        # exog: future values of X from start to end
        return self.results.predict(start=start, end=end, exog=exog)


class GaussianProcessModel:
    def __init__(self, kernel=None):
        if kernel is None:
            # Standard kernel for time series: RBF + noise
            self.kernel = C(1.0) * RBF(length_scale=1.0) + C(0.1)
        else:
            self.kernel = kernel
        
        self.model = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=0.1,  # Noise level
            normalize_y=True,
            n_restarts_optimizer=10
        )
        
    def train(self, features, target):
        """Train the GP model on historical data"""
        self.model.fit(features, target)
        return self
    
    def predict(self, features_future, return_std=True):
        """Generate forecasts with uncertainty intervals"""
        if return_std:
            mean_pred, std_pred = self.model.predict(features_future, return_std=True)
            return mean_pred, std_pred
        else:
            return self.model.predict(features_future)

import numpy as np

class XGBoostModel:

    def __init__(self, params=None):

        self.xgboost = xgb.XGBRegressor(enable_categorical=True)

    def train(self, x, y):


        #drop NaNs and Infs
        y = y.dropna()
        x = x.loc[y.index]

        self.xgboost.fit(x, y)


            
    def predict(self, x):

        return self.xgboost.predict(x)
    
import lightgbm as lgb
import pandas as pd
import numpy as np

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


class CatBoostModel:
    pass


from pygam import LinearGAM, s, f
import numpy as np
import pandas as pd

class GAMModel:
    def __init__(self, params=None):
        # You can set default parameters or pass them via 'params'
        self.params = params or {}
        # Here we create a simple LinearGAM with smooth terms.
        # Adjust the spline basis (s) or factor (f) terms as needed based on your features.
        # For example, if you have 10 features, you might initialize:
        # self.gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9), **self.params)
        # For a generic implementation, we delay setting up the terms until training.
        self.gam = None

    def train(self, x: pd.DataFrame, y: pd.Series):
        # Drop NaNs and Infs in target and align features
        y = y.dropna()
        x = x.loc[y.index]

        n_features = x.shape[1]
        # Create a sum of smooth splines for each feature.
        term_list = [s(i) for i in range(n_features)]
        # Initialize with the first term, then add the rest
        terms = term_list[0]
        for term in term_list[1:]:
            terms += term
        # Initialize the GAM model with the specified terms and additional parameters.
        self.gam = LinearGAM(terms, **self.params)
        
        # Fit the model on your training data.
        self.gam.fit(x.values, y.values)

    def predict(self, x: pd.DataFrame):
        # Predict using the trained GAM model.
        if self.gam is None:
            raise ValueError("Model has not been trained yet.")
        return self.gam.predict(x.values)
