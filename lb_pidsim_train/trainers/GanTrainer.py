#from __future__ import annotations

import os
import tensorflow as tf
import matplotlib.pyplot as plt

from lb_pidsim_train.trainers import TensorTrainer


TF_FLOAT = tf.float32
"""Default data-type for tensors."""


class GanTrainer (TensorTrainer):
  def _training_plots(self, report, history):
    plt.figure (figsize = (8,5), dpi = 100)
    plt.plot (history.history["d_loss"], linewidth=1.5, color="dodgerblue", label = "discriminator")
    plt.plot (history.history["g_loss"], linewidth=1.5, color="coral", label = "generator")
    plt.legend (loc="upper right")
    report.add_figure(); plt.clf(); plt.close()

    plt.figure (figsize = (8,5), dpi = 100)
    plt.plot (history.history["kl_div"], linewidth=1.5, color="forestgreen", label = "K-L divergence")
    plt.legend (loc="upper right")
    report.add_figure(); plt.clf(); plt.close()

    plt.figure (figsize = (8,5), dpi = 100)
    plt.plot (history.history["d_lr"], linewidth=1.5, color="dodgerblue", label = "d_lr")
    plt.plot (history.history["g_lr"], linewidth=1.5, color="coral", label = "g_lr")
    plt.legend (loc="center right")
    report.add_figure(); plt.clf(); plt.close()

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
