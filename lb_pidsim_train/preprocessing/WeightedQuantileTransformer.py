import numpy as np
from sklearn.preprocessing import QuantileTransformer


class WeightedQuantileTransformer (QuantileTransformer):
  def _compute_weight_distortion_1d(self, var, sample_weights):
    indices = np.argsort (var)
    cumulative = np.cumsum(sample_weights[indices])
    cumulative /= cumulative[-1]
    return np.interp(self.references_, cumulative, var[indices])

  def fit (self, X, y=None, sample_weights=None):
    QuantileTransformer.fit(self, X, y)
    if sample_weights is not None:
      self.quantiles_ = np.array([
          self._compute_weight_distortion_1d(x, np.asarray(sample_weights)) 
            for x, q in zip(X.T, self.quantiles_.T)
          ]).T
    return self
