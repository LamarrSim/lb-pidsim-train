from __future__ import annotations

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def name_not_passed (attr_name, default = None) -> str:
  """Warning message for dirname or filename not passed.
  
  Parameters
  ----------
  attr_name : `str`
    Dirname or filename not passed.
    
  default : `str`, optional
    Default attribute name assigned.
    
  Returns
  -------
  message : `str`
    Warning message.

  Examples
  --------
  >>> from lb-pidsim-train.core.trainers.utils.warn_message import name_not_passed
  >>> mess = name_not_passed ("export dirname", "./results")
  >>> print (mess)
  No export dirname passed, './results' will be assigned by default.
  """
  message = "No %s passed, '%s' will be assigned by default." % (attr_name, default)
  return message


def directory_not_found (dirname) -> str:
  """Warning message for directory not found.
  
  Parameters
  ----------
  dirname : `str`
    Name of the directory not found.

  Returns
  -------
    message : `str`
      Warning message.

  Examples
  --------
  >>> from lb-pidsim-train.core.trainers.utils.warn_message import directory_not_found
  >>> mess = directory_not_found ("./results")
  >>> print (mess)
  './results' not found, the directory will be created.
  """
  message = "'%s' not found, the directory will be created." % dirname
  return message
  