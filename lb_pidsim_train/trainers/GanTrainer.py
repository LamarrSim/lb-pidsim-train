#from __future__ import annotations

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from lb_pidsim_train.trainers import TensorTrainer


NP_FLOAT = np.float32
"""Default data-type for arrays."""

TF_FLOAT = tf.float32
"""Default data-type for tensors."""


class GanTrainer (TensorTrainer):   # TODO class description
  def _training_plots (self, report, history) -> None:   # TODO complete docstring
    """short description
    
    Parameters
    ----------
    report : ...
      ...

    history : ...
      ...

    See Also
    --------
    html_reports.Report : ...
      ...
    """
    plt.figure (figsize = (8,5), dpi = 100)
    plt.title  ("Learning curves", fontsize = 14)   # TODO plot loss variance
    plt.xlabel ("Training epochs", fontsize = 12)
    plt.ylabel (f"{self.model.loss_name}", fontsize = 12)
    plt.plot (history.history["d_loss"], linewidth = 1.5, color = "dodgerblue", label = "discriminator")
    plt.plot (history.history["g_loss"], linewidth = 1.5, color = "coral", label = "generator")
    plt.legend (title = "Training players:", loc = "upper right", fontsize = 10)
    report.add_figure(); plt.clf(); plt.close()

    plt.figure (figsize = (8,5), dpi = 100)
    plt.title  ("Metric curves", fontsize = 14)
    plt.xlabel ("Training epochs", fontsize = 12)
    plt.ylabel ("Mean square error", fontsize = 12)
    plt.plot (history.history["mse"], linewidth = 1.5, color = "forestgreen", label = "training set")
    if self._validation_split != 0.0:
      plt.plot (history.history["val_mse"], linewidth = 1.5, color = "orangered", label = "validation set")
    plt.legend (loc = "upper right", fontsize = 10)
    report.add_figure(); plt.clf(); plt.close()

    plt.figure (figsize = (8,5), dpi = 100)
    plt.title  ("Learning rate scheduling", fontsize = 14)
    plt.xlabel ("Training epochs", fontsize = 12)
    plt.ylabel ("Learning rate", fontsize = 12)
    plt.plot (history.history["d_lr"], linewidth = 1.5, color = "dodgerblue", label = "discriminator")
    plt.plot (history.history["g_lr"], linewidth = 1.5, color = "coral", label = "generator")
    plt.legend (title = "Training players:", loc = "center right", fontsize = 10)
    report.add_figure(); plt.clf(); plt.close()

  def _save_model ( self, name, model, verbose = False ) -> None:   # TODO fix docstring
    """Save the trained generator.
    
    Parameters
    ----------
    name : `str`
      Name of the directory containing the TensorFlow SavedModel file.

    model : `tf.keras.Model`
      GAN model taken from `lb_pidsim_train.algorithms.gan` and configured 
      for the training procedure.

    verbose : `bool`, optional
      Verbosity mode. `False` = silent (default), `True` = a control message 
      is printed. 

    See Also
    --------
    lb_pidsim_train.algorithms.gan :
      ...

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

  def generate (self, X) -> np.ndarray:   # TODO complete docstring
    """Method to generate the target variables `Y` given the input features `X`.
    
    Parameters
    ----------
    X : `np.ndarray` or `tf.Tensor`
      ...

    Returns
    -------
    Y : `np.ndarray`
      ...
    """
    ## Data-type control
    if isinstance (X, np.ndarray):
      X = tf.convert_to_tensor ( X, dtype = TF_FLOAT )
    elif isinstance (X, tf.Tensor):
      X = tf.cast (X, dtype = TF_FLOAT)
    else:
      TypeError ("error")  # TODO insert error message

    ## Sample random points in the latent space
    batch_size = tf.shape(X)[0]
    latent_dim = self.model.latent_dim
    latent_tensor = tf.random.normal ( shape = (batch_size, latent_dim), dtype = TF_FLOAT )

    ## Map the latent space into the generated space
    input_tensor = tf.concat ( [X, latent_tensor], axis = 1 )
    Y = self.model.generator (input_tensor) 
    Y = Y.numpy() . astype (NP_FLOAT)   # casting to numpy array
    return Y

  @property
  def discriminator (self) -> tf.keras.Sequential:
    """The discriminator after the training procedure."""
    return self.model.discriminator

  @property
  def generator (self) -> tf.keras.Sequential:
    """The generator after the training procedure."""
    return self.model.generator



if __name__ == "__main__":   # TODO complete __main__
  trainer = GanTrainer ( "test", export_dir = "./models", report_dir = "./reports" )
  trainer . feed_from_root_files ( "../data/Zmumu.root", ["px1", "py1", "pz1"], "E1" )
  print ( trainer.datachunk.describe() )
