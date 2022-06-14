#from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, FunctionTransformer
from lb_pidsim_train.preprocessing import WeightedQuantileTransformer, LbColTransformer

STRATEGIES = ["pass-through", "minmax", "standard", "simple-quantile", "weighted-quantile"]


def preprocessor ( data ,
                   weights = None ,
                   strategies = "standard", 
                   cols_to_transform = None ) -> LbColTransformer:
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
  scaler : `lb_pidsim_train.utils.LbColTransformer`
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
  [[ 3.73251479  1.80495666  1.74969503  1.        ]
   [ 0.06708618 -0.43481269  2.85217265  1.        ]
   [ 1.02627701 -2.477204    2.94474338  1.        ]
   ...
   [-4.11890958  2.12049264  2.40337247 -1.        ]
   [ 2.8872164  -4.50525599  1.35950868  1.        ]
   [ 4.76668602 -1.1865085   0.65532981  1.        ]]
  >>> w = np.random.normal ( size = 1000 )
  >>> from lb_pidsim_train.utils import preprocessor
  >>> scaler = preprocessor ( data, weights = w, strategies = ["standard","minmax"], cols_to_transform = [(0,1),2] )
  >>> data_scaled = scaler . transform (data)
  >>> print (data_scaled)
  [[ 1.12664787  0.7559853   0.14892489  1.        ]
   [-0.06086264  0.18719734  0.24302712  1.        ]
   [ 0.24989207 -0.3314666   0.25092852  1.        ]
   ...
   [-1.41702434  0.83611546  0.20471968 -1.        ]
   [ 0.85279157 -0.84648909  0.11562044  1.        ]
   [ 1.46169442 -0.00369532  0.05551509  1.        ]]
  >>> data_inv_tr = scaler . inverse_transform (data_scaled)
  >>> print (data_inv_tr)
  [[ 3.73251479  1.80495666  1.74969503  1.        ]
   [ 0.06708618 -0.43481269  2.85217265  1.        ]
   [ 1.02627701 -2.477204    2.94474338  1.        ]
   ...
   [-4.11890958  2.12049264  2.40337247 -1.        ]
   [ 2.8872164  -4.50525599  1.35950868  1.        ]
   [ 4.76668602 -1.1865085   0.65532981  1.        ]]
  >>> err = np.max (abs (data_inv_tr - data) / (1 + abs (data)))
  >>> print (err)
  2.329972186166122e-16
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
    elif "-".join(strategy.split("-")[:2]) == "simple-quantile":
      if len(strategy.split("-")) == 3:
        n_quantiles = int ( strategy.split("-")[2] )
      else:
        n_quantiles = 1000 
      scaler = QuantileTransformer ( n_quantiles = n_quantiles , 
                                     subsample = int (1e8) ,
                                     output_distribution = "normal" )
    elif "-".join(strategy.split("-")[:2]) == "weighted-quantile":
      if len(strategy.split("-")) == 3:
        n_quantiles = int ( strategy.split("-")[2] )
      else:
        n_quantiles = 1000 
      scaler = WeightedQuantileTransformer ( n_quantiles = n_quantiles , 
                                             subsample = int (1e8) ,
                                             output_distribution = "normal" )
    else:
      raise ValueError ( f"Preprocessing strategy not implemented. Available strategies are " 
                         f"{STRATEGIES}, '{strategy}' passed." )

    transformers . append ( (strategy.replace("-","_"), scaler, col) )
    scaled_cols += col
    del scaler

  scaled_cols = np.unique (scaled_cols)
  cols_to_ignore = list ( np.delete (indices, scaled_cols) )

  if len(cols_to_ignore) > 0:
    final_scaler = LbColTransformer ( transformers + \
                                        [ ( "pass_through", FunctionTransformer(), cols_to_ignore ) ] 
                                      )
  else:
    final_scaler = LbColTransformer ( transformers )
  
  final_scaler . fit ( data, sample_weight = weights )
  return final_scaler



if __name__ == "__main__":
  ## Dataset
  a = np.random.uniform ( -5, 5, size = (1000,2) )
  b = np.random.exponential ( 2, size = 1000 )
  c = np.where (a[:,0] < 0, -1, 1)
  w = np.random.normal ( size = 1000 )
  data = np.c_ [a, b, c]
  print ("\t\t\t\t\t+-----------+")
  print ("\t\t\t\t\t|   DATA    |")
  print ("\t\t\t\t\t+-----------+")
  print (data, "\n")
  # print (w, "\n")

  ## Dataset after preprocessing
  scaler = preprocessor ( data, weights = w, strategies = ["standard","minmax"], cols_to_transform = [(0,1),2] )
  data_scaled = scaler . transform (data)
  print ("\t\t\t\t\t+--------------------+")
  print ("\t\t\t\t\t|   FIT_TRANSFORM    |")
  print ("\t\t\t\t\t+--------------------+")
  print (data_scaled, "\n")

  ## Dataset back-projected
  data_inv_tr = scaler . inverse_transform (data_scaled)
  print ("\t\t\t\t\t+------------------------+")
  print ("\t\t\t\t\t|   INVERSE_TRANSFORM    |")
  print ("\t\t\t\t\t+------------------------+")
  print (data_inv_tr, "\n")

  err = np.max (abs (data_inv_tr - data) / (1 + abs (data)))
  print ("\t\t\t\t\t+------------+")
  print ("\t\t\t\t\t|   ERROR    |")
  print ("\t\t\t\t\t+------------+")
  print (err)
