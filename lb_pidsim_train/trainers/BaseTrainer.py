#from __future__ import annotations

import os
import time
import pickle
import uproot
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

from datetime import datetime
from warnings import warn
from sklearn.utils import shuffle

import lb_pidsim_train as pidsim
from lb_pidsim_train.utils import warn_message as wm
from lb_pidsim_train.utils import data_from_trees, nan_filter, preprocessor

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


TF_FLOAT = tf.float32
"""Default data-type for tensors."""

NP_FLOAT = TF_FLOAT.as_numpy_dtype
"""Default data-type for arrays."""


class BaseTrainer:
  """Base class for training generative models.
  
  Parameters
  ----------
  name : `str`
    Name of the trained model.

  export_dir : `str`, optional
    Export directory for the trained model.

  export_name : `str`, optional
    Export file name for the trained model.

  report_dir : `str`, optional
    Report directory for the trained model.

  report_name : `str`, optional
    Report file name for the trained model.
  """
  def __init__ ( self ,
                 name ,
                 export_dir  = None ,
                 export_name = None ,
                 report_dir  = None ,
                 report_name = None ) -> None:

    timestamp = str (datetime.now()) . split (".") [0]
    timestamp = timestamp . replace (" ","_")
    version = ""
    for time, unit in zip ( timestamp.split(":"), ["h","m","s"] ):
      version += time + unit   # YYYY-MM-DD_HHhMMmSSs

    self._name = name

    if export_dir is None:
      export_dir = "./results"
      message = wm.name_not_passed ("export dirname", export_dir)
      warn (message)
    self._export_dir = export_dir
    if not os.path.exists (self._export_dir):
      message = wm.directory_not_found (self._export_dir)
      warn (message)
      os.makedirs (self._export_dir)

    if export_name is None:
      export_name = "{}_{}" . format (name, version)
      message = wm.name_not_passed ("export filename", export_name)
      warn (message)
    self._export_name = export_name

    if report_dir is None:
      report_dir = "./reports"
      message = wm.name_not_passed ("report dirname", report_dir)
      warn (message)
    self._report_dir = report_dir
    if not os.path.exists (self._report_dir):
      message = wm.directory_not_found (self._report_dir)
      warn (message)
      os.makedirs (self._report_dir)

    if report_name is None:
      report_name = "{}_{}" . format (name, version)
      message = wm.name_not_passed ("report filename", report_name)
      warn (message)
    self._report_name = report_name

#  def feed_with_dataframes ( self ,
#                             dataframes , 
#                             X_vars , 
#                             Y_vars ,
#                             w_var  = None ,
#                             selections = None ) -> None:
#    """Feed the training procedure with dataframes.
#    
#    Parameters
#    ----------
#    dataframes : `pd.DataFrame` or `list` of `pd.DataFrame`
#      List of dataframes used for the training procedure.
#
#    X_vars : `str` or `list` of `str`
#      Column names of the input variables within the dataframes.
#
#    Y_vars : `str` or `list` of `str`
#      Column names of the output variables within the dataframes.
#    
#    w_var : `str` or `list` of `str`, optional
#      Column name of the weight variable, if available, within the 
#      dataframes (`None`, by default).
#
#    selections : `str` or `list` of `str`, optional
#      Boolean expressions to filter the dataframes (`None`, by default).
#    """
#    ## List data-type promotion
#    if isinstance (dataframes, pd.DataFrame):
#      dataframes = [dataframes]
#    if isinstance (X_vars, str):
#      X_vars = [X_vars]
#    if isinstance (Y_vars, str):
#      Y_vars = [Y_vars]
#    if isinstance (w_var, str):
#      w_var = [w_var]
#    if isinstance (selections, str):
#      selections = [selections]
#
#    self._X_vars = X_vars
#    self._Y_vars = Y_vars
#    self._w_var  = w_var
#
#    ## List of column names
#    if w_var is not None:
#      cols = X_vars + Y_vars + w_var
#    else:
#      cols = X_vars + Y_vars 
#  
#    ## Dataframes combination
#    data = pd.concat (dataframes, ignore_index = True)
#    data = data[cols]
#
#    ## Data selection
#    if selections:
#      queries = "&".join ("(%s)" % s for s in selections)
#      data.query (queries, inplace = True)
#
#    self._datachunk = data
#    self._more_data_avail = False

  def feed_from_root_files ( self ,
                             root_files  , 
                             X_vars = None , 
                             Y_vars = None ,
                             w_var  = None ,
                             selections = None ,
                             tree_names = None ,
                             chunk_size = None ,
                             verbose = 0 ) -> None:
    """Feed the training procedure with ROOT files.
    
    Parameters
    ----------
    root_files : `str` or `list` of `str`
      List of ROOT files used for the training procedure.

    X_vars : `str` or `list` of `str`, optional
      Branch names of the input variables within the ROOT trees
      (`None`, by default).

    Y_vars : `str` or `list` of `str`, optional
      Branch names of the output variables within the ROOT trees
      (`None`, by default).
    
    w_var : `str` or `list` of `str`, optional
      Branch name of the weight variable, if available, within the 
      ROOT trees (`None`, by default).

    selections : `str` or `list` of `str`, optional
      Boolean expressions to filter the ROOT trees (`None`, by default).

    tree_names : `str` or `list` of `str`, optional
      If more than one ROOT tree is defined for each file, the ones to 
      be loaded have to be defined specifying their names as the keys 
      (`None`, by default).

    chunk_size : `int` or `list` of `int`, optional
      Total number of instance rows loaded to disk for the training 
      procedure (`None`, by default).

    verbose : {0, 1}, optional
      Verbosity mode. `0` = silent (default), `1` = time for data-chunk 
      loading is printed. 

    See Also
    --------
    lb_pidsim_train.utils.data_from_trees :
      Stratified data shuffling from list of `uproot.TTree`.
    """
    ## List data-type promotion
    if isinstance (root_files, str):
      root_files = [root_files]
    if isinstance (X_vars, str):
      X_vars = [X_vars]
    if isinstance (Y_vars, str):
      Y_vars = [Y_vars]
    if isinstance (w_var, str):
      w_var = [w_var]
    if isinstance (selections, str):
      selections = [selections]
    if isinstance (tree_names, str):
      tree_names = [tree_names]

    self._X_vars = X_vars
    self._Y_vars = Y_vars
    self._w_var  = w_var

    ## List of branch names
    branches = list()
    if X_vars is not None:
      branches += X_vars
      if Y_vars is not None:
        branches += Y_vars
      if w_var is not None:
        branches += w_var
    else:
      branches = None

    ## Length match
    if tree_names is None:
      tree_names = [ None for i in range ( len(root_files) ) ]

    ## Check files and tree names match
    if len(root_files) != len(tree_names):
      raise ValueError ("The number of ROOT files should match with the tree names passed.")

    ## ROOT trees extraction
    trees = list()
    for fname, tname in zip (root_files, tree_names):
      file = uproot.open (fname)
      if tname is not None:
        key = tname
      else:
        key = file.keys()
        key = key[0] . split (";") [0]   # take the tree name
      t = file [key]
      trees . append (t)

    ## Data selection
    if selections:
      selections = "&".join ("(%s)" % s for s in selections)

    start = time.time()
    self._datachunk = data_from_trees ( trees = trees , 
                                        branches = branches ,
                                        cut = selections    ,
                                        chunk_size = chunk_size )
    stop = time.time()
    if (verbose > 0): print ( f"Data-chunk correctly loaded in {stop-start:.3f} s" )

  def prepare_dataset ( self ,
                        X_preprocessing = None ,
                        Y_preprocessing = None ,
                        X_vars_to_preprocess = None ,
                        Y_vars_to_preprocess = None ,
                        subsample_size = 100000 ,
                        verbose = 0 ) -> None:
    """Split the data-chunk into X, Y and w, and perform preprocessing.

    Parameters
    ----------
    X_preprocessing : {None, 'minmax', 'standard', 'quantile'}, optional
      Preprocessing strategy for the input-set. The choices are `None` 
      (default), `'minmax'`, `'standard'` and `'quantile'`. If `None` is
      selected, no preprocessing is performed at all.

    Y_preprocessing : {None, 'minmax', 'standard', 'quantile'}, optional
      Preprocessing strategy for the output-set. The choices are `None` 
      (default), `'minmax'`, `'standard'` and `'quantile'`. If `None` is
      selected, no preprocessing is performed at all.

    X_vars_to_preprocess : `str` or `list` of `str`, optional
      List of input variables to preprocess (`None`, by default). If `None` 
      is selected, all the input variables are preprocessed.

    Y_vars_to_preprocess : `str` or `list` of `str`, optional
      List of output variables to preprocess (`None`, by default). If `None` 
      is selected, all the output variables are preprocessed.

    subsample_size : `int`, optional
      Data-chunk subsample size used to compute the preprocessing transformer 
      parameters (`100000`, by default).

    verbose : {0, 1, 2}, optional
      Verbosity mode. `0` = silent (default), `1` = time for preprocessing 
      steps is printed, `2`= also time for shuffling is printed together
      with control messages. 

    See Also
    --------
    lb_pidsim_train.utils.preprocessor :
      Scikit-Learn transformer for data preprocessing.
    """
    X, Y, w = self._unpack_data()
    start = time.time()
    X, Y, w = shuffle (X, Y, w)
    stop = time.time()
    if (verbose > 1): print ( f"Shuffle-time: {stop-start:.3f} s" )

    ## Weight array
    self._w = w

    ## Data-type control
    try:
      subsample_size = int ( subsample_size )
    except:
      raise TypeError ("The sub-sample size should be an integer.")

    ## Preprocessed input array
    if X_preprocessing is not None:
      start = time.time()
      if X_vars_to_preprocess is not None:
        X_cols_to_preprocess = list()
        for idx, var in enumerate (self._X_vars):
          if var in X_vars_to_preprocess:
            X_cols_to_preprocess . append (idx)   # column index
      else:
        X_cols_to_preprocess = None
      scaler_X = preprocessor ( X[:subsample_size], strategy = X_preprocessing, 
                                cols_to_transform = X_cols_to_preprocess )
      self._X  = scaler_X . transform (X)   # transform the input-set
      stop = time.time()
      if (verbose > 0): print ( f"Preprocessing time for X: {stop-start:.3f} s" )
      self._save_transformer ( "transform_X", scaler_X, verbose = (verbose > 1) )
    else:
      self._X = X

    ## Preprocessed output array
    if Y_preprocessing is not None:
      start = time.time()
      if Y_vars_to_preprocess is not None:
        Y_cols_to_preprocess = list()
        for idx, var in enumerate (self._Y_vars):
          if var in Y_vars_to_preprocess:
            Y_cols_to_preprocess . append (idx)   # column index
      else:
        Y_cols_to_preprocess = None
      scaler_Y = preprocessor ( Y[:subsample_size], strategy = Y_preprocessing, 
                                cols_to_transform = Y_cols_to_preprocess )
      self._Y  = scaler_Y . transform (Y)   # transform the output-set
      stop = time.time()
      if (verbose > 0): print ( f"Preprocessing time for Y: {stop-start:.3f} s" )
      self._save_transformer ( "transform_Y", scaler_Y, verbose = (verbose > 1) )
    else:
      self._Y = Y

  def _unpack_data (self) -> tuple:
    """Unpack the data-chunk into input, output and weights 
    (array of ones, if not available).

    See Also
    --------
    lb_pidsim_train.utils.nan_filter : 
      Clean arrays from NaN elements.
    """
    ## Input array
    if self._X_vars is not None:
      X = nan_filter ( self._datachunk[self._X_vars] . to_numpy() )
    else:
      X = nan_filter ( self._datachunk . to_numpy() )

    ## Output array
    if self._Y_vars is not None:
      Y = nan_filter ( self._datachunk[self._Y_vars] . to_numpy() )
    else:
      raise ValueError ("No variables have been passed to create an output-set.")

    ## Weight array
    if self._w_var is not None:
      w = self._datachunk[self._w_var] . to_numpy()
    else:
      w = np.ones ( X.shape[0], dtype = NP_FLOAT )

    X . astype ( NP_FLOAT )
    Y . astype ( NP_FLOAT )
    return X, Y, w

  def _save_transformer (self, name, transformer, verbose = False) -> None:
    """Save the preprocessing transformer.
    
    Parameters
    ----------
    name : `str`
      Name of the pickle file containing the transformer.

    transformer : `ColumnTransformer`
      Preprocessing transformer resulting from `lb_pidsim_train.utils.preprocessor`.

    verbose : `bool`, optional
      Verbosity mode. `False` = silent (default), `True` = a control message is printed. 

    See Also
    --------
    lb_pidsim_train.utils.preprocessor :
      Scikit-Learn transformer for data preprocessing.
    """
    dirname = f"{self._export_dir}/{self._export_name}"
    if not os.path.exists (dirname):
      os.makedirs (dirname)
    filename = f"{dirname}/{name}.pkl"
    pickle . dump ( transformer, open (filename, "wb") )
    if verbose: print ( f"Transformer correctly exported to {filename}" )
       
  @property
  def name (self) -> str:
    """Name of the trained model."""
    return self._name

  @property
  def X_vars (self) -> list:
    """Names of the input variables (`None`, if not available)."""
    return self._X_vars

  @property
  def Y_vars (self) -> list:
    """Names of the output variables (`None`, if not available)."""
    return self._Y_vars

  @property
  def w_var (self) -> list:
    """Name of the weight variable (`None`, if not available)."""
    return self._w_var

  @property
  def datachunk (self) -> pd.DataFrame:
    """Dataset used for the training procedure."""
    return self._datachunk

  @property
  def X (self) -> np.ndarray:
    """Array containing the input-set after preprocessing."""
    return self._X

  @property
  def Y (self) -> np.ndarray:
    """Array containing the output-set after preprocessing."""
    return self._Y

  @property
  def w (self) -> np.ndarray:
    """Array containing the weights (array of ones, if not available)."""
    return self._w

    

if __name__ == "__main__":
  trainer = BaseTrainer ("test", "./generators", "test", "./reports", "test")
  trainer . feed_from_root_files ("../data/Zmumu.root", ["px1", "py1", "pz1"], "E1")
  print (trainer.datachunk.describe())
