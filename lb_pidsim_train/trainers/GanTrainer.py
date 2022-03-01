#from __future__ import annotations

import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from time import time
from sklearn.utils import shuffle
from lb_pidsim_train.trainers import TensorTrainer
from lb_pidsim_train.utils import PidsimColTransformer


NP_FLOAT = np.float32
"""Default data-type for arrays."""

TF_FLOAT = tf.float32
"""Default data-type for tensors."""


class GanTrainer (TensorTrainer):   # TODO class description
  def load_model ( self , 
                   filepath , 
                   model_to_load = "all" ,
                   save_transformer = True ,
                   verbose = 0 ) -> None:   # TODO add docstring
    """"""
    if not self._datachunk_filled:
      raise RuntimeError ("error")   # TODO implement error message
    
    if self._dataset_prepared:
      raise RuntimeError ("error")   # TODO implement error message

    if model_to_load not in ["gen", "disc", "all"]:
      raise ValueError ("`model_to_save` should be chosen in ['gen', 'disc', 'all'].")

    ## Unpack data
    X, Y, w = self._unpack_data()
    start = time()
    X, Y, w = shuffle (X, Y, w)
    stop = time()
    if verbose: print ( f"Shuffle-time: {stop-start:.3f} s" )

    self._X = X
    self._Y = Y
    self._w = w

    ## Preprocessed input array
    file_X = f"{filepath}/transform_X.pkl"
    if os.path.exists (file_X):
      start = time()
      self._scaler_X = PidsimColTransformer ( pickle.load (open (file_X, "rb")) )
      if (verbose > 0):
        print (f"Transformer correctly loaded from {file_X}.")
      self._X_scaled = self._scaler_X . transform ( self.X )
      stop = time()
      if (verbose > 1):
        print (f"Preprocessing time for X: {stop-start:.3f} s")
      if save_transformer: 
        self._save_transformer ( "transform_X" , 
                                 self._scaler_X.sklearn_transformer ,   # saved as Scikit-Learn class
                                 verbose = (verbose > 0) )
    else:
      self._scaler_X = None
      self._X_scaled = self.X

    ## Preprocessed output array
    file_Y = f"{filepath}/transform_Y.pkl"
    if os.path.exists (file_Y):
      start = time()
      self._scaler_Y = PidsimColTransformer ( pickle.load (open (file_Y, "rb")) )
      if (verbose > 0):
        print (f"Transformer correctly loaded from {file_Y}.")
      self._Y_scaled = self._scaler_Y . transform ( self.Y )
      stop = time()
      if (verbose > 1):
        print (f"Preprocessing time for Y: {stop-start:.3f} s")
      if save_transformer:
        self._save_transformer ( "transform_Y" , 
                                 self._scaler_Y.sklearn_transformer ,   # saved as Scikit-Learn class 
                                 verbose = (verbose > 0) )
    else:
      self._scaler_Y = None
      self._Y_scaled = self.Y

    ## Load the models
    if model_to_load == "gen":
      self._generator = tf.keras.models.load_model (f"{filepath}/saved_generator")
      self._gen_loaded = True
    elif model_to_load == "disc":
      self._discriminator = tf.keras.models.load_model (f"{filepath}/saved_discriminator")
      self._disc_loaded = True
    else:
      self._generator = tf.keras.models.load_model (f"{filepath}/saved_generator")
      self._discriminator = tf.keras.models.load_model (f"{filepath}/saved_discriminator")
      self._gen_loaded = self._disc_loaded = True
    self._model_loaded = True
  
  def extract_model ( self, player = "gen", fine_tuned_layers = None ) -> list:   # TODO add docstring
    """"""
    if player == "gen":
      if not self._gen_loaded:
        raise RuntimeError ("error")   # TODO implement error message
      model = self._generator
    elif player == "disc":
      if not self._disc_loaded:
        raise RuntimeError ("error")   # TODO implement error message
      model = self._discriminator
    else:
      raise ValueError ("error")   # TODO implement error message

    num_layers = len ( model.layers[:-1] )

    ## Data-type control
    if fine_tuned_layers is not None:
      try:
        fine_tuned_layers = int ( fine_tuned_layers )
      except:
        raise TypeError (f"The number of layers to fine-tune should be an integer," 
                         f" instead {type(fine_tuned_layers)} passed." )
    else:
      fine_tuned_layers = num_layers

    layers = list()
    for i, layer in enumerate ( model.layers[:-1] ):
      layer._name = f"loaded_{layer.name}"
      if i < (num_layers - fine_tuned_layers): 
        layer.trainable = False
      else:
        layer.trainable = True
      layers . append (layer)

    return layers

  def train_model ( self , 
                    model , 
                    batch_size = 1 , 
                    num_epochs = 1 , 
                    validation_split = 0.0 , 
                    scheduler = None , 
                    verbose = 0 ) -> None:
    super().train_model ( model = model , 
                          batch_size = 2 * batch_size , 
                          num_epochs = num_epochs , 
                          validation_split = validation_split , 
                          scheduler = scheduler , 
                          verbose = verbose )

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
    n_epochs = len (history.history["mse"])

    ## Metric curves plots
    plt.figure (figsize = (8,5), dpi = 100)
    plt.title  ("Metric curves", fontsize = 14)
    plt.xlabel ("Training epochs", fontsize = 12)
    plt.ylabel ("Mean square error", fontsize = 12)
    plt.plot (history.history["mse"], linewidth = 1.5, color = "forestgreen", label = "training set")
    if self._validation_split != 0.0:
      plt.plot (history.history["val_mse"], linewidth = 1.5, color = "orangered", label = "validation set")
    plt.legend (loc = "upper right", fontsize = 10)
    y_bottom = min ( min(history.history["mse"][int(n_epochs/10):]), min(history.history["val_mse"][int(n_epochs/10):]) )
    y_top    = max ( max(history.history["mse"][int(n_epochs/10):]), max(history.history["val_mse"][int(n_epochs/10):]) )
    y_bottom -= 0.1 * y_bottom
    y_top    += 0.1 * y_top
    plt.ylim (bottom = y_bottom, top = y_top)

    report.add_figure(); plt.clf(); plt.close()

    ## Learning curves plots
    plt.figure (figsize = (8,5), dpi = 100)
    plt.title  ("Learning curves", fontsize = 14)   # TODO plot loss variance
    plt.xlabel ("Training epochs", fontsize = 12)
    plt.ylabel (f"{self.model.loss_name}", fontsize = 12)
    plt.plot (history.history["d_loss"], linewidth = 1.5, color = "dodgerblue", label = "discriminator (train-set)")
    if self._validation_split != 0.0:
      plt.plot (history.history["val_d_loss"], linewidth = 1.5, color = "seagreen", label = "discriminator (val-set)")
    plt.plot (history.history["g_loss"], linewidth = 1.5, color = "coral", label = "generator (train-set)")
    if self._validation_split != 0.0:
      plt.plot (history.history["val_g_loss"], linewidth = 1.5, color = "orange", label = "generator (val-set)")
    plt.legend (title = "Adversarial players:", loc = "upper right", fontsize = 10)
    y_bottom = min ( min(history.history["d_loss"][int(n_epochs/10):]), min(history.history["g_loss"][int(n_epochs/10):]) )
    y_top    = max ( max(history.history["d_loss"][int(n_epochs/10):]), max(history.history["g_loss"][int(n_epochs/10):]) )
    y_bottom += 0.1 * y_bottom
    y_top    += 0.1 * y_top
    plt.ylim (bottom = y_bottom, top = y_top)

    report.add_figure(); plt.clf(); plt.close()

    ## Learning rate scheduling plots
    plt.figure (figsize = (8,5), dpi = 100)
    plt.title  ("Learning rate scheduling", fontsize = 14)
    plt.xlabel ("Training epochs", fontsize = 12)
    plt.ylabel ("Learning rate", fontsize = 12)
    plt.plot (history.history["d_lr"], linewidth = 1.5, color = "dodgerblue", label = "discriminator")
    plt.plot (history.history["g_lr"], linewidth = 1.5, color = "coral", label = "generator")
    plt.yscale ("log")
    plt.legend (title = "Adversarial players:", loc = "upper right", fontsize = 10)

    report.add_figure(); plt.clf(); plt.close()

    ## Correlation plots
    Y_ref  = self.Y
    Y_gen  = self._scaler_Y . inverse_transform ( self.generate (self.X_scaled) )

    for i, y_var in enumerate (self.Y_vars):
      fig = plt.figure ( figsize = (20, 6), dpi = 200 )
      gs = gridspec.GridSpec ( nrows = 2 , 
                               ncols = 4 ,
                               wspace = 0.25 ,
                               hspace = 0.25 ,
                               width_ratios  = [2, 1, 1, 1] , 
                               height_ratios = [1, 1] )

      ax = fig.add_subplot ( gs[0:,0] )
      ax . set_xlabel (y_var, fontsize = 12)
      ax . set_ylabel ("Candidates", fontsize = 12)
      _, bins, _ = ax . hist (Y_ref[:,i], bins = 100, density = True, weights = self.w, color = "dodgerblue", label = "Original")
      ax . hist (Y_gen[:,i], bins = bins, density = True, histtype = "step", color = "deeppink", label = "Generated")
      ax . legend (loc = "upper left", fontsize = 10)

      ax_p_ref = fig.add_subplot ( gs[0,1] )
      ax_p_ref . set_xlabel (y_var, fontsize = 10)
      ax_p_ref . set_ylabel ("Momentum [Gev/$c$]", fontsize = 10)
      _, binx_p, biny_p, _ = ax_p_ref . hist2d (Y_ref[:,i], self.X[:,0]/1e3, bins = 25, density = True, weights = self.w, cmin = 0)
      ax_p_ref . annotate ( "original", color = "w", weight = "bold",
                            ha = "center", va = "center", size = 10,
                            xy = (0.8, 0.9), xycoords = "axes fraction", 
                            bbox = dict (boxstyle = "round", fc = "dodgerblue", alpha = 1.0, ec = "1.0") )

      ax_p_gen = fig.add_subplot ( gs[1,1] )
      ax_p_gen . set_xlabel (y_var, fontsize = 10)
      ax_p_gen . set_ylabel ("Momentum [Gev/$c$]", fontsize = 10)
      ax_p_gen . hist2d (Y_gen[:,i], self.X[:,0]/1e3, bins = [binx_p, biny_p], density = True)
      ax_p_gen . annotate ( "generated", color = "w", weight = "bold",
                            ha = "center", va = "center", size = 10,
                            xy = (0.8, 0.9), xycoords = "axes fraction", 
                            bbox = dict (boxstyle = "round", fc = "deeppink", alpha = 1.0, ec = "1.0") )

      ax_eta_ref = fig.add_subplot ( gs[0,2] )
      ax_eta_ref . set_xlabel (y_var, fontsize = 10)
      ax_eta_ref . set_ylabel ("Pseudorapidity", fontsize = 10)
      _, binx_eta, biny_eta, _ = ax_eta_ref . hist2d (Y_ref[:,i], self.X[:,1], bins = 25, density = True, weights = self.w, cmin = 0)
      ax_eta_ref . annotate ( "original", color = "w", weight = "bold",
                              ha = "center", va = "center", size = 10,
                              xy = (0.8, 0.9), xycoords = "axes fraction", 
                              bbox = dict (boxstyle = "round", fc = "dodgerblue", alpha = 1.0, ec = "1.0") )

      ax_eta_gen = fig.add_subplot ( gs[1,2] )
      ax_eta_gen . set_xlabel (y_var, fontsize = 10)
      ax_eta_gen . set_ylabel ("Pseudorapidity", fontsize = 10)
      ax_eta_gen . hist2d (Y_gen[:,i], self.X[:,1], bins = [binx_eta, biny_eta], density = True)
      ax_eta_gen . annotate ( "generated", color = "w", weight = "bold",
                              ha = "center", va = "center", size = 10,
                              xy = (0.8, 0.9), xycoords = "axes fraction", 
                              bbox = dict (boxstyle = "round", fc = "deeppink", alpha = 1.0, ec = "1.0") )

      ax_ntk_ref = fig.add_subplot ( gs[0,3] )
      ax_ntk_ref . set_xlabel (y_var, fontsize = 10)
      ax_ntk_ref . set_ylabel ("$\mathtt{nTracks}$", fontsize = 10)
      _, binx_ntk, biny_ntk, _ = ax_ntk_ref . hist2d (Y_ref[:,i], self.X[:,2], bins = 25, density = True, weights = self.w, cmin = 0)
      ax_ntk_ref . annotate ( "original", color = "w", weight = "bold",
                              ha = "center", va = "center", size = 10,
                              xy = (0.8, 0.9), xycoords = "axes fraction", 
                              bbox = dict (boxstyle = "round", fc = "dodgerblue", alpha = 1.0, ec = "1.0") )

      ax_ntk_gen = fig.add_subplot ( gs[1,3] )
      ax_ntk_gen . set_xlabel (y_var, fontsize = 10)
      ax_ntk_gen . set_ylabel ("$\mathtt{nTracks}$", fontsize = 10)
      ax_ntk_gen . hist2d (Y_gen[:,i], self.X[:,2], bins = [binx_ntk, biny_ntk], density = True)
      ax_ntk_gen . annotate ( "generated", color = "w", weight = "bold",
                              ha = "center", va = "center", size = 10,
                              xy = (0.8, 0.9), xycoords = "axes fraction", 
                              bbox = dict (boxstyle = "round", fc = "deeppink", alpha = 1.0, ec = "1.0") )

      report.add_figure(); plt.clf(); plt.close()

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
