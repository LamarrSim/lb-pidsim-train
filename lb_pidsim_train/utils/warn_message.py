#from __future__ import annotations


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
  >>> from lb_pidsim_train.utils.warn_message import name_not_passed
  >>> mess = name_not_passed ("export dirname", "./models")
  >>> print (mess)
  No export dirname passed, './models' will be assigned by default.
  """
  message = f"[WARNING] No {attr_name} passed, '{default}' will be assigned by default."
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
  >>> from lb_pidsim-train.utils.warn_message import directory_not_found
  >>> mess = directory_not_found ("./models")
  >>> print (mess)
  './models' not found, the directory will be created.
  """
  message = f"[WARNING] '{dirname}' not found, the directory will be created."
  return message
  