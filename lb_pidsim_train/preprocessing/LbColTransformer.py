#from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer

STRATEGIES = ["pass_through", "minmax", "standard", "simple_quantile", "weighted_quantile"]


class LbColTransformer:   # TODO add docstring
  def __init__ (self, *args, **kwargs):

    if (len(args) == 1) and isinstance (args[0], ColumnTransformer):
      self._col_transformer = args[0]
    else:
      self._col_transformer = ColumnTransformer (*args, **kwargs)

  def fit (self, X, y = None, sample_weight = None):
    if sample_weight is None:
      return self._col_transformer.fit (X, y = y)

    else:
      transformers = list()

      ## ColumnTransformer stuff
      self._col_transformer . _check_n_features (X, reset = True)
      self._col_transformer . _validate_transformers()
      self._col_transformer . _validate_column_callables (X)
      self._col_transformer . _validate_remainder (X)
      self._col_transformer . _validate_transformers()
      self._col_transformer . _validate_column_callables (X)

      for transf in self._col_transformer.transformers:
        name, scaler, cols = transf
        if name in ["standard", "weighted_quantile"]:   # sample_weight supported
          scaler . fit ( X = X[:,cols], sample_weight = sample_weight )
        else:
          scaler . fit ( X = X[:,cols] )
        transformers . append ( scaler )   # fitted transformers

      self._col_transformer . _update_fitted_transformers (transformers)
      return self._col_transformer

  def fit_transform (self, X, y = None, sample_weight = None):
    if sample_weight is None:
      return self._col_transformer.fit_transform (X, y = y)
    else:
      return self . fit (X, y = y, sample_weight = sample_weight) . transform (X)

  def transform (self, X):
    # return self._col_transformer.transform (X)
    return self._transform (X, inverse = False)

  def inverse_transform (self, X):
    return self._transform (X, inverse = True)

  def _transform (self, X, inverse = False):
    X_tr = np.empty (shape = X.shape)   # initial array

    ## Transformers: ( (name, fitted_transformer, column) , ... )
    for transf in self._col_transformer.transformers_:
      name, scaler, cols = transf
      if name == "pass_through":
        X_tr[:,cols] = X[:,cols]
      else:
        if inverse:
          X_tr[:,cols] = scaler . inverse_transform (X[:,cols]) 
        else:
          X_tr[:,cols] = scaler . transform (X[:,cols])      

    return X_tr

  @property
  def sklearn_transformer (self) -> ColumnTransformer:
    return self._col_transformer
