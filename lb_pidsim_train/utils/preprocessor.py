#from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, FunctionTransformer
from sklearn.compose       import ColumnTransformer


def preprocessor ( data ,
                   strategy = "quantile" , 
                   cols_to_transform = None ) -> ColumnTransformer:
  """Scikit-Learn transformer for data preprocessing.
  
  Parameters
  ----------
  data : `np.ndarray`
    Array to preprocess according to a specific strategy.

  strategy : {'quantile', 'standard', 'minmax'}, optional
    Strategy to use for preprocessing (`'quantile'`, by default).
    The `'quantile'` strategy relies on the Scikit-Learn's 
    `QuantileTransformer`, `'standard'` implements `StandardScaler`,
    while `'minmax'` stands for `MinMaxScaler`.

  cols_to_transform : `list` of `int` or `np.ndarray`, optional
    Indices of the data columns to which apply the preprocessing 
    transformation (`None`, by default). If `None` is selected, 
    all the data columns are preprocessed.

  Returns
  -------
  scaler : `ColumnTransformer`
    Scikit-Learn transformer fitted and ready to use (calling the 
    `transform` method).

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
  >>> a = np.random.uniform (-5, 5, 1000)
  >>> b = np.random.exponential (2, 1000)
  >>> c = np.where (a < 0, -1, 1)
  >>> data = np.c_ [a, b, c]
  >>> print (data)
  [[-2.73768881  2.40009296 -1.        ]
   [ 2.90420319  2.97191117  1.        ]
   [-3.54111824  3.56929996 -1.        ]
   ...
   [-1.89415656  1.00781813 -1.        ]
   [-1.88903141  3.10907516 -1.        ]
   [-3.23689531  1.37872621 -1.        ]]
  >>> from lb_pidsim_train.utils import preprocessor
  >>> scaler = preprocessor (data, "quantile", [0,1])
  >>> data_scaled = scaler . transform (data)
  >>> print (data_scaled)
  [[-0.73808885  0.52066149 -1.        ]
   [ 0.83023416  0.7347981   1.        ]
   [-1.0887225   0.98969684 -1.        ]
   ...
   [-0.47802469 -0.29398578 -1.        ]
   [-0.46960306  0.79532164 -1.        ]
   [-0.93390994 -0.03136936 -1.        ]]
  """
  ## Strategy selection
  if strategy == "minmax":
    num_scaler = MinMaxScaler()
  elif strategy == "standard":
    num_scaler = StandardScaler()
  elif strategy == "quantile":
    num_scaler = QuantileTransformer ( output_distribution = "normal" )
  else:
    raise ValueError ( f"Preprocessing strategy not implemented. Available strategies " 
                       f"are ['quantile', 'standard', 'minmax'], '{strategy}' passed." )

  ## Default column indices
  all_cols = np.arange (data.shape[1]) . astype (np.int32)
  if cols_to_transform is not None:
    cols_to_transform = list ( cols_to_transform )
    if len(cols_to_transform) != 0:
      cols_to_ignore = list ( np.delete (all_cols, cols_to_transform) )
    else:
      cols_to_ignore = list ( all_cols )
  else:
    cols_to_transform = list ( all_cols )
    cols_to_ignore = []

  scaler = ColumnTransformer ( [
                                 ( "num", num_scaler, cols_to_transform ) ,
                                 ( "cls", FunctionTransformer(), cols_to_ignore )
                               ] )
  
  scaler . fit ( data )
  return scaler



if __name__ == "__main__":
  ## Dataset
  a = np.random.uniform (-5, 5, 1000)
  b = np.random.exponential (2, 1000)
  c = np.where (a < 0, -1, 1)
  data = np.c_ [a, b, c]
  print (data)

  ## Dataset after preprocessing
  scaler = preprocessor (data, "quantile", [0,1])
  data_scaled = scaler . transform (data)
  print (data_scaled)
