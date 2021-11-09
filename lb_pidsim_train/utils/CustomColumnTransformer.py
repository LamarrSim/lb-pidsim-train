#from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer


class CustomColumnTransformer (ColumnTransformer):
  def inverse_transform (self, X):
    """Back-projection to the original space.
    
    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
      Input data, of which specified subsets are used to fit the transformers.

    Returns
    -------
    X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
      Transformed array.
    """
    X_tr = np.empty (shape = X.shape)   # initial array

    ## Transformers: ( (name, fitted_transformer, column) , ... )
    transformers = self.transformers_

    ## Numerical transformer
    num_transformer = transformers[0][1]
    num_cols = transformers[0][2]
    X_tr[:,num_cols] = num_transformer . inverse_transform (X[:,num_cols])

    ## Function transformer
    fnc_cols = transformers[1][2]
    X_tr[:,fnc_cols] = X[:,fnc_cols]

    return X_tr
