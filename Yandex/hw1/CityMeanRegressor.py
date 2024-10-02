import numpy as np
from sklearn.base import RegressorMixin


class CityMeanRegressor(RegressorMixin):
    def fit(self, X=None, y=None):
        self.mskMean = np.mean(y[X.index[X.city == "msk"]])
        self.spbMean = np.mean(y[X.index[X.city == "spb"]])

    def predict(self, X=None):
        ans = np.zeros((X.shape[0],))
        ans[np.array(X.city == "msk")] = self.mskMean
        ans[np.array(X.city == "spb")] = self.spbMean
        return ans