#from __future__ import annotations

import os
import tensorflow as tf

from lb_pidsim_train.trainers import TensorTrainer


TF_FLOAT = tf.float32
"""Default data-type for tensors."""


class GanTrainer (TensorTrainer):
  def _save_model ( self, name, model, verbose = False ) -> None:
    """Save the trained model.
    
    Parameters
    ----------
    name : `str`
      Name of the directory containing the TensorFlow SavedModel file.

    model : `tf.keras.Model`
      TensorFlow model configured for the training procedure.

    verbose : `bool`, optional
      Verbosity mode. `False` = silent (default), `True` = a control message is printed. 

    See Also
    --------
    tf.keras.Model :
      Set of layers with training and inference features.

    tf.keras.models.save_model :
      Save a model as a TensorFlow SavedModel or HDF5 file.
    """
    dirname = f"{self._export_dir}/{self._export_name}"
    if not os.path.exists (dirname):
      os.makedirs (dirname)
    filename = f"{dirname}/{name}"
    model.generator . save ( f"{filename}/saved_model", save_format = "tf" )
    if verbose: print ( f"Trained generator correctly exported to {filename}" )
