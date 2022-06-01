#from __future__ import annotations

import numpy as np

STRATEGIES = ["pass-through", "minmax", "standard", "simple-quantile", "weighted-quantile"]

def pre_preprocessing_step (transformers, vars_to_transform, all_vars) -> tuple:   # TODO add docstring
  ## Data-type control
  if isinstance (transformers, str):
    if "-".join (transformers.split("-")[:2]) not in STRATEGIES:
        raise ValueError ( f"Preprocessing strategy not implemented. Available strategies " 
                           f"are {STRATEGIES}, '{transformers}' passed." )
    transformers = [transformers]
    vars_to_transform = [vars_to_transform]

  elif isinstance (transformers, list):
    for preprocess in transformers:
      if "-".join (preprocess.split("-")[:2]) not in STRATEGIES:
        raise ValueError ( f"Preprocessing strategy not implemented. Available strategies " 
                           f"are {STRATEGIES}, '{preprocess}' passed." )
    if not isinstance (vars_to_transform, list):
      raise ValueError ( f"The list of variables to preprocess should be passed as a list "
                         f"of length equal to the preprocessing stategies one." )

  ## Check length matching
  if len(transformers) != len(vars_to_transform):
    raise ValueError ("Transformers and variables to transform length don't match.")

  ## Get column indices
  cols_to_transform = list()
  for idx, var in enumerate (all_vars):
    if var in vars_to_transform:
      cols_to_transform . append (idx)   # column indices

  ## Prepare list of variables to transform
  arranged_vars = list()
  arranged_cols = list()
  unique_transformers = np.unique (transformers)
  unique_transformers = unique_transformers[unique_transformers != "pass-through"]

  ## Loop to fill variables to transform
  for u_transf in unique_transformers:
    tmp_vars = list()
    tmp_cols = list()
    for transf, col in zip (transformers, cols_to_transform):
      if transf == u_transf:
        tmp_vars . append (all_vars[col])   # variable to transform
        tmp_cols . append (col)        # column to transform
    arranged_vars . append ( tuple(tmp_vars) )
    arranged_cols . append ( tuple(tmp_cols) )
            
  return list(unique_transformers), arranged_vars, arranged_cols
