#from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer


class PidsimColTransformer:   # TODO add docstring
  def __init__ (self, *args, **kwargs):

    if (len(args) == 1) and isinstance (args[0], ColumnTransformer):
      self._col_transformer = args[0]
    else:
      self._col_transformer = ColumnTransformer (*args, **kwargs)

  def fit (self, *args, **kwargs):
    self._col_transformer.fit (*args, **kwargs)

  def transform (self, *args, **kwargs):
    return self._col_transformer.transform (*args, **kwargs)

  def fit_transform (self, *args, **kwargs):
    return self._col_transformer.fit_transform (*args, **kwargs)

  def inverse_transform (self, X):
    X_tr = np.empty (shape = X.shape)   # initial array

    ## Transformers: ( (name, fitted_transformer, column) , ... )
    transformers = self._col_transformer.transformers_

    ## Numerical transformers
    for ct in transformers[:-1]:
      num_transf, num_cols = ct[1], ct[2]
      X_tr[:,num_cols] = num_transf . inverse_transform (X[:,num_cols])

    ## Pass-through transformer
    pt_cols = transformers[-1][2]
    X_tr[:,pt_cols] = X[:,pt_cols]

    return X_tr

  @property
  def sklearn_transformer (self) -> ColumnTransformer:
    return self._col_transformer
