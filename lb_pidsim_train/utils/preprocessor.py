#from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, FunctionTransformer
from lb_pidsim_train.utils import PidsimColTransformer


def preprocessor ( data ,
                   strategies , 
                   cols_to_transform = None ) -> PidsimColTransformer:
  """Scikit-Learn transformer for data preprocessing.
  
  Parameters
  ----------
  data : `np.ndarray`
    Array to preprocess according to a specific strategy.

  strategies : {'quantile-highbin', 'quantile-lowbin', 'standard', 'minmax'}, list of strategies
    Strategy to use for preprocessing (`'quantile-highbin'`, by default).
    The `'quantile-*'` strategies rely on the Scikit-Learn's 
    `QuantileTransformer`, `'standard'` implements `StandardScaler`,
    while `'minmax'` stands for `MinMaxScaler`.

  cols_to_transform : `tuple` or `list` of `tuple`, optional
    Indices of the data columns to which apply the preprocessing 
    transformation (`None`, by default). If `None` is selected, 
    all the data columns are preprocessed.

  Returns
  -------
  scaler : `lb_pidsim_train.utils.PidsimColTransformer`
    Scikit-Learn transformer fitted and ready to use (calling 
    the `transform` method).

  See Also
  --------
  sklearn.preprocessing.QuantileTransformer :
    Transform features using quantiles information.

  sklearn.preprocessing.StandardScaler :
    Standardize features by removing the mean and scaling to unit variance.

  sklearn.preprocessing.MinMaxScaler :
    Transform features by scaling each feature to a given range.

  Examples
  --------
  >>> import numpy as np
  >>> a = np.random.uniform ( -5, 5, size = (1000,2) )
  >>> b = np.random.exponential ( 2, size = 1000 )
  >>> c = np.where ( a[:,0] < 0, -1, 1 )
  >>> data = np.c_ [a, b, c]
  >>> print (data)
  [[ 4.48592375  2.58189332  0.33288739  1.        ]
   [-0.18804986 -0.72362047  1.3272863  -1.        ]
   [ 3.97474628  1.00386848  0.01257831  1.        ]
   ...
   [ 3.59323927 -4.89095396  1.39652514  1.        ]
   [ 0.01236004  2.96300772  2.00442855  1.        ]
   [ 0.30250978  4.52430715  1.07108719  1.        ]]
  >>> from lb_pidsim_train.utils import preprocessor
  >>> scaler = preprocessor ( data, ["standard","minmax"], [(0,1),2] )
  >>> data_scaled = scaler . transform (data)
  >>> print (data_scaled)
  [[ 1.57515219e+00  9.07773146e-01  2.10757643e-02  1.00000000e+00]
   [-2.73322282e-02 -2.42265881e-01  8.59144492e-02 -1.00000000e+00]
   [ 1.39989362e+00  3.58754061e-01  1.90363833e-04  1.00000000e+00]
   ...
   [ 1.26909292e+00 -1.69214522e+00  9.04290914e-02  1.00000000e+00]
   [ 4.13788463e-02  1.04036870e+00  1.30066764e-01  1.00000000e+00]
   [ 1.40857465e-01  1.58356876e+00  6.92092689e-02  1.00000000e+00]]
  >>> data_inv_tr = scaler . inverse_transform (data_scaled)
  >>> print (data_inv_tr)
  [[ 4.48592375  2.58189332  0.33288739  1.        ]
   [-0.18804986 -0.72362047  1.3272863  -1.        ]
   [ 3.97474628  1.00386848  0.01257831  1.        ]
   ...
   [ 3.59323927 -4.89095396  1.39652514  1.        ]
   [ 0.01236004  2.96300772  2.00442855  1.        ]
   [ 0.30250978  4.52430715  1.07108719  1.        ]]
  >>> err = np.max (abs (data_inv_tr - data) / (1 + abs (data)))
  >>> print (err)
  1.7737186817999972e-16
  """
  ## List data-type promotion
  if isinstance (strategies, str):
    strategies = [strategies]
  if isinstance (cols_to_transform, tuple):
    cols_to_transform = [cols_to_transform]

  ## Default column indices
  indices = np.arange (data.shape[1]) . astype (np.int32)
  if cols_to_transform is None:
    cols_to_transform = [ tuple (indices) ]
    cols_to_ignore = tuple()

  ## Length matching 
  if len(strategies) != len(cols_to_transform):
    raise ValueError ( f"The list of strategies ({len(strategies)}) and the column "
                       f"indices ({len(cols_to_transform)}) provided don't match." )

  transformers = list()
  scaled_cols  = list()

  ## Preprocessor per column
  for strategy, col in zip (strategies, cols_to_transform):
    if isinstance (col, int):
      col = [col]
    elif isinstance (col, tuple):
      col = list(col)

    if strategy == "minmax":
      scaler = MinMaxScaler()
    elif strategy == "standard":
      scaler = StandardScaler()
    elif strategy == "quantile-highbin":
      scaler = QuantileTransformer ( n_quantiles = 1000 , 
                                     subsample = int (1e8) ,
                                     output_distribution = "normal" )
    elif strategy == "quantile-lowbin":
      scaler = QuantileTransformer ( n_quantiles = 10 , 
                                     subsample = int (1e8) ,
                                     output_distribution = "normal" )    
    else:
      raise ValueError ( f"Preprocessing strategy not implemented. Available strategies are " 
                         f"['quantile-highbin', 'quantile-lowbin', 'standard', 'minmax'], "
                         f"'{strategy}' passed." )

    transformers . append ( (strategy.replace("-","_"), scaler, col) )
    scaled_cols += col
    del scaler

  scaled_cols = np.unique (scaled_cols)
  cols_to_ignore = list ( np.delete (indices, scaled_cols) )

  final_scaler = PidsimColTransformer ( transformers + \
                                        [ ( "pass-through", FunctionTransformer(), cols_to_ignore ) ] 
                                      )
  
  final_scaler . fit ( data )
  return final_scaler



if __name__ == "__main__":
  ## Dataset
  a = np.random.uniform ( -5, 5, size = (1000,2) )
  b = np.random.exponential ( 2, size = 1000 )
  c = np.where (a[:,0] < 0, -1, 1)
  data = np.c_ [a, b, c]
  print (data)

  ## Dataset after preprocessing
  scaler = preprocessor ( data, ["standard","minmax"], [(0,1),2] )
  data_scaled = scaler . transform (data)
  print (data_scaled)

  ## Dataset back-projected
  data_inv_tr = scaler . inverse_transform (data_scaled)
  print (data_inv_tr)

  err = np.max (abs (data_inv_tr - data) / (1 + abs (data)))
  print (err)
