from sklearn.linear_model import LinearRegression

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import xgboost as xgb

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

from statsmodels.tsa.arima.model import ARIMA

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



class XGBoostModel:

    def __init__(self, params=None):

        self.xgboost = xgb.XGBRegressor()

    def train(self, x, y):
        self.xgboost.fit(x, y)
    
    def predict(self, x):
        return self.xgboost.predict(x)