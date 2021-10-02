from __future__ import annotations

import os
import uproot
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

from datetime import datetime
from warnings import warn

import lb_pidsim_train as pidsim
from lb_pidsim_train.utils import warn_message as wm
from lb_pidsim_train.utils import data_from_trees, nan_filter

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
    for time, unit in zip (timestamp.split(":"), ["h","m","s"]):
      version += time + unit   # YYYY-MM-DD_HHdMMmSSs

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

  def feed_with_dataframes ( self ,
                             dataframes  , 
                             input_vars  , 
                             output_vars ,
                             weight_var  = None ,
                             selections  = None ) -> None:
    """Feed the training procedure with dataframes.
    
    Parameters
    ----------
    dataframes : `pd.DataFrame` or `list` of `pd.DataFrame`
      List of dataframes used for the training procedure.

    input_vars : `str` or `list` of `str`
      Column names of the input variables within the dataframes.

    output_vars : `str` or `list` of `str`
      Column names of the output variables within the dataframes.
    
    weight_var : `str` or `list` of `str`, optional
      Column name of the weight variable, if available, within the 
      dataframes (`None`, by default).

    selections : `str` or `list` of `str`, optional
      Boolean expressions to filter the dataframes (`None`, by default).
    """
    ## List data-type promotion
    if isinstance (dataframes, pd.DataFrame):
      dataframes = [dataframes]
    if isinstance (input_vars, str):
      input_vars = [input_vars]
    if isinstance (output_vars, str):
      output_vars = [output_vars]
    if isinstance (weight_var, str):
      weight_var = [weight_var]
    if isinstance (selections, str):
      selections = [selections]

    self._input_vars  = input_vars
    self._output_vars = output_vars
    self._weight_var  = weight_var

    ## List of column names
    if weight_var:
      cols = input_vars + output_vars + weight_var
    else:
      cols = input_vars + output_vars 
  
    ## Dataframes combination
    data = pd.concat (dataframes, ignore_index = True)
    data = data[cols]

    ## Data selection
    if selections:
      queries = "&".join ("(%s)" % s for s in selections)
      data.query (queries, inplace = True)

    self._datachunk = data
    self._more_data_avail= False

  def feed_from_root_files ( self ,
                             root_files  , 
                             input_vars  , 
                             output_vars ,
                             weight_var  = None ,
                             selections  = None ,
                             tree_names  = None ,
                             chunk_size  = 1000 ,
                             max_nfiles  = 100  ) -> None:
    """Feed the training procedure with ROOT files.
    
    Parameters
    ----------
    root_files : `str` or `list` of `str`
      List of ROOT files used for the training procedure.

    input_vars : `str` or `list` of `str`
      Branch names of the input variables within the ROOT trees.

    output_vars : `str` or `list` of `str`
      Branch names of the output variables within the ROOT trees.
    
    weight_var : `str` or `list` of `str`, optional
      Branch name of the weight variable, if available, within the 
      ROOT trees (`None`, by default).

    selections : `str` or `list` of `str`, optional
      Boolean expressions to filter the ROOT trees (`None`, by default).

    tree_names : `str` or `list` of `str`, optional
      If more than one ROOT tree is defined for each file, the ones to 
      be loaded have to be defined specifying their names as the keys 
      (`None`, by default).

    chunk_size : `int` or `list` of `int`, optional
      Total number of instance rows to be loaded to disk at once (`1000`, 
      by default).

    max_nfiles : `int` or `list` of `int`, optional
      Maximum number of files from which to pick instances for training.
      Can be decreased if files are similar to each other and accessing 
      too often to all of them affects performance (`10`, by default).
    """
    ## List data-type promotion
    if isinstance (root_files, str):
      root_files = [root_files]
    if isinstance (input_vars, str):
      input_vars = [input_vars]
    if isinstance (output_vars, str):
      output_vars = [output_vars]
    if isinstance (weight_var, str):
      weight_var = [weight_var]
    if isinstance (selections, str):
      selections = [selections]
    if isinstance (tree_names, str):
      tree_names = [tree_names]

    self._input_vars  = input_vars
    self._output_vars = output_vars
    self._weight_var  = weight_var
    
    ## Data-type control
    try:
      chunk_size = int ( chunk_size )
    except:
      ValueError ("The chunk-size should be an integer.")
    try:
      max_nfiles = int ( max_nfiles )
    except:
      ValueError ("The maximum number of files should be an integer.")

    self._chunk_size = chunk_size
    self._max_nfiles = max_nfiles

    ## List of branch names
    if weight_var:
      branches = input_vars + output_vars + weight_var
    else:
      branches = input_vars + output_vars 

    ## Length match
    if tree_names is None:
      tree_names = [ None for i in range (len(root_files)) ]

    ## Check files and tree names match
    if len(root_files) != len(tree_names):
      raise ValueError ("The number of ROOT files should match with the tree names passed.")

    ## ROOT trees extraction
    trees = list()
    for fname, tname in zip (root_files, tree_names):
      file = uproot.open (fname)
      if tname:
        key = tname
      else:
        key = file.keys()
        key = key[0] . split (";") [0]   # take the tree name
      t = file [key]
      trees . append (t)

    ## Data selection
    if selections:
      selections = "&".join ("(%s)" % s for s in selections)

    self._selections = selections

    data = data_from_trees ( trees = trees , 
                             branches = branches ,
                             cut = selections    ,
                             max_ntrees = max_nfiles ,
                             chunk_size = chunk_size )

    self._datachunk = data
    self._all_data  = False

  def _load_data (self, dtype = NP_FLOAT) -> tuple:
    """"description
    
    Parameters
    ----------
    dtype : `np.dtype`, optional
      ...
    """
    X = nan_filter ( np.stack ( [ self._datachunk[v] for v in self._input_vars  ] ) ) . T
    Y = nan_filter ( np.stack ( [ self._datachunk[v] for v in self._output_vars ] ) ) . T
    self._data_rows = len (X)

    if self._weight_var:
      w =  np.c_ [ self._datachunk[self._weight_var] ]
    else:
      w =  np.ones ( X.shape[0], dtype = dtype )

    X = X.astype ( dtype )
    Y = Y.astype ( dtype )
    w = w.astype ( dtype )
    return X, Y, w
       
  @property
  def name (self) -> str:
    """Name of the trained model."""
    return self._name

  @property
  def input_vars (self) -> list:
    """Names of the input variables."""
    return self._input_vars

  @property
  def output_vars (self) -> list:
    """Names of the output variables."""
    return self._output_vars

  @property
  def weight_var (self) -> list or None:
    """Name of the weight variable (if available)."""
    return self._weight_var

  @property
  def datachunk (self) -> pd.DataFrame:
    """Dataframe containing the current dataset (it may change during the training)."""
    return self._datachunk

    




if __name__ == "__main__":
  trainer = BaseTrainer ("test", "./generators", "test", "./reports", "test")
  trainer . feed_from_root_files ("../data/Zmumu.root", ["px1", "py1", "pz1"], "E1")
  print (trainer.datachunk.describe())