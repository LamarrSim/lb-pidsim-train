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

  strategies : {'minmax', 'standard', 'simple-quantile', 'weighted-quantile'}, list of strategies
    Strategy to use for preprocessing (`'standard'`, by default).
    The `'*-quantile'` strategies rely on the Scikit-Learn's 
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
  >>> b = np.random.exponential ( 2, size = (1000,2) )
  >>> c = np.where ( a[:,0] < 0, -1, 1 )
  >>> data = np.c_ [a, b, c]
  >>> print (data)
  [[-1.07596365  3.13736563  0.91456348  6.88416941 -1.        ]
   [-1.85459971  4.2439101   1.90906239  2.97599951 -1.        ]
   [-1.31761994 -2.83872989  1.07036459  1.97196161 -1.        ]
   ...
   [-1.84121035 -3.86157199  2.7373931   0.67573819 -1.        ]
   [ 2.79119745 -0.48778006  0.27411022  1.01966811  1.        ]
   [ 3.09058781 -4.26678205  1.43145045  1.70428456  1.        ]]
  >>> w = np.random.normal ( size = 1000 )
  >>> from lb_pidsim_train.utils import preprocessor
  >>> scaler = preprocessor ( data, weights = w, strategies = ["standard","minmax"], cols_to_transform = [(0,2),3] )
  >>> data_scaled = scaler . transform (data)
  >>> print (data_scaled)
  [[20.29739155 -2.20668287  0.44542308  3.13736563 -1.        ]
   [19.5187555  -1.88849891  0.19237571  4.2439101  -1.        ]
   [20.05573526 -2.15683524  0.12736596 -2.83872989 -1.        ]
   ...
   [19.53214486 -1.62347948  0.04343769 -3.86157199 -1.        ]
   [24.16455265 -2.41159204  0.06570657 -0.48778006  1.        ]
   [24.46394302 -2.04130799  0.11003432 -4.26678205  1.        ]]
  >>> data_inv_tr = scaler . inverse_transform (data_scaled)
  >>> print (data_inv_tr)
  [[-1.07596365  3.13736563  0.91456348  6.88416941 -1.        ]
   [-1.85459971  4.2439101   1.90906239  2.97599951 -1.        ]
   [-1.31761994 -2.83872989  1.07036459  1.97196161 -1.        ]
   ...
   [-1.84121035 -3.86157199  2.7373931   0.67573819 -1.        ]
   [ 2.79119745 -0.48778006  0.27411022  1.01966811  1.        ]
   [ 3.09058781 -4.26678205  1.43145045  1.70428456  1.        ]]
  >>> err = np.max (abs (data_inv_tr - data) / (1 + abs (data)))
  >>> print (err)
  1.7370222821991345e-15
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
  b = np.random.exponential ( 2, size = (1000,2) )
  c = np.where (a[:,0] < 0, -1, 1)
  w = np.random.normal ( size = 1000 )
  data = np.c_ [a, b, c]
  print ("\t\t\t\t\t+-----------+")
  print ("\t\t\t\t\t|   DATA    |")
  print ("\t\t\t\t\t+-----------+")
  print (data, "\n")
  # print (w, "\n")

  ## Dataset after preprocessing
  scaler = preprocessor ( data, weights = w, strategies = ["standard","minmax"], cols_to_transform = [(0,2),3] )
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
