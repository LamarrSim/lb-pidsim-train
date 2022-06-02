#from __future__ import annotations

import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from time import time
from sklearn.utils import shuffle
from tensorflow.keras.models       import Sequential
from tensorflow.keras.layers       import Dense, LeakyReLU
from tensorflow.keras.optimizers   import RMSprop
from tensorflow.keras.losses       import MeanSquaredError
from lb_pidsim_train.trainers      import TensorTrainer
from lb_pidsim_train.preprocessing import LbColTransformer
from lb_pidsim_train.metrics       import KS_test


NP_FLOAT = np.float32
"""Default data-type for arrays."""

TF_FLOAT = tf.float32
"""Default data-type for tensors."""


class GanTrainer (TensorTrainer):   # TODO class description

  def prepare_dataset ( self , 
                        X_preprocessing = None , 
                        Y_preprocessing = None , 
                        X_vars_to_preprocess = None , 
                        Y_vars_to_preprocess = None , 
                        subsample_size = 500000 , 
                        enable_reweights = True ,
                        save_transformer = True , 
                        verbose = 0 ) -> None:
    super().prepare_dataset ( X_preprocessing = X_preprocessing , 
                              Y_preprocessing = Y_preprocessing , 
                              X_vars_to_preprocess = X_vars_to_preprocess , 
                              Y_vars_to_preprocess = Y_vars_to_preprocess , 
                              subsample_size = subsample_size , 
                              save_transformer = save_transformer , 
                              verbose = verbose )

    if enable_reweights:
      if self.w_var is not None:
        reweighter = self._train_reweighter ( num_epochs = 10 ,
                                              batch_size = 1024 ,
                                              save_model = save_transformer ,
                                              verbose = verbose )
        with tf.device ("/cpu:0"): 
          X = tf.cast ( tf.convert_to_tensor(self.X_scaled) , dtype = TF_FLOAT )
          self._w_X = reweighter(X) . numpy() . reshape(self._w_Y.shape) . astype(NP_FLOAT)
      else:
        print ( "[WARNING] No reweighting strategy available, since there aren't weights to reweight" )

    self._rw_enabled = enable_reweights

  def _train_reweighter ( self ,
                          num_epochs = 1 ,
                          batch_size = None ,
                          save_model = True ,
                          verbose = 0 ) -> tf.keras.Sequential:   # TODO add docstring
    physical_devices = tf.config.list_physical_devices ("GPU")

    ## Memory allocation
    if ( len (physical_devices) > 0 ):
      with tf.device ("/gpu:0"):
        input  = tf.cast ( tf.convert_to_tensor(self.X_scaled) , dtype = TF_FLOAT )
        output = tf.cast ( tf.convert_to_tensor(self.w)        , dtype = TF_FLOAT )
    else:
      with tf.device ("/cpu:0"):
        input  = tf.cast ( tf.convert_to_tensor(self.X_scaled) , dtype = TF_FLOAT )
        output = tf.cast ( tf.convert_to_tensor(self.w)        , dtype = TF_FLOAT )

    ## TF dataset
    dataset = tf.data.Dataset.from_tensor_slices ( (input, output) )
    dataset = dataset.batch ( batch_size, drop_remainder = True )
    dataset = dataset.cache()
    dataset = dataset.prefetch ( tf.data.AUTOTUNE )

    ## Model construction
    reweighter = Sequential()
    for layer in range (5):
      reweighter.add ( Dense (units = 32) )
      reweighter.add ( LeakyReLU (alpha = 0.05) )
    reweighter.add ( Dense (units = 1, activation = "relu") )

    ## Model configuration
    reweighter.compile ( optimizer = RMSprop (learning_rate = 5e-4), loss = MeanSquaredError() )

    ## Model training
    start = time()
    rw_history = reweighter . fit ( dataset, epochs = num_epochs, verbose = 0 )
    stop = time()
    if (verbose > 0): 
      print ( f"[INFO] Reweighter training completed in {(stop-start)/60:.3f} min" )
    if (verbose > 1):
      ks_test = KS_test ( x_obs = self.X_scaled , 
                          x_exp = self.X_scaled , 
                          w_obs = reweighter(self.X_scaled) . numpy() . flatten() , 
                          w_exp = self.w . flatten() )
      print ( f"[INFO] Worst reweighter performance: {max(ks_test):.4f} (K-S test)" )

    ## Model saving
    if save_model:
      filename = f"{self._export_dir}/{self._export_name}/saved_reweighter"
      if not os.path.exists (filename): os.makedirs (filename)
      reweighter.save (filename, save_format = "tf")
      if (verbose > 0): print ( f"[INFO] Reweighter correctly exported to {filename}" )
    return reweighter

  def load_model ( self , 
                   filepath , 
                   model_to_load = "all" ,
                   enable_reweights = True ,
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
    if verbose: print ( f"[INFO] Whole data-chunk shuffled in {stop-start:.3f} s" )

    self._X = X
    self._Y = Y
    self._w = w

    ## Preprocessed input array
    file_X = f"{filepath}/transform_X.pkl"
    if os.path.exists (file_X):
      start = time()
      self._scaler_X = LbColTransformer ( pickle.load (open (file_X, "rb")) )
      if (verbose > 0): print ( f"[INFO] Transformer correctly loaded from {file_X}" )
      self._X_scaled = self._scaler_X . transform ( self.X )
      stop = time()
      if (verbose > 1): print ( f"[INFO] X-features preprocessed in {stop-start:.3f} s" )
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
      self._scaler_Y = LbColTransformer ( pickle.load (open (file_Y, "rb")) )
      if (verbose > 0): print ( f"[INFO] Transformer correctly loaded from {file_Y}" )
      self._Y_scaled = self._scaler_Y . transform ( self.Y )
      stop = time()
      if (verbose > 1): print ( f"[INFO] Y-features preprocessed in {stop-start:.3f} s" )
      if save_transformer:
        self._save_transformer ( "transform_Y" , 
                                 self._scaler_Y.sklearn_transformer ,   # saved as Scikit-Learn class 
                                 verbose = (verbose > 0) )
    else:
      self._scaler_Y = None
      self._Y_scaled = self.Y

    ## Weights or reweighted weights
    self._w_X = np.copy (self._w)
    self._w_Y = np.copy (self._w)
    if enable_reweights:
      if self.w_var is not None:
        reweighter = self._train_reweighter ( num_epochs = 10 ,
                                              batch_size = 1024 ,
                                              save_model = save_transformer ,
                                              verbose = verbose )
        with tf.device ("/cpu:0"): 
          X = tf.cast ( tf.convert_to_tensor(self.X_scaled) , dtype = TF_FLOAT )
          self._w_X = reweighter(X) . numpy() . reshape(self._w_Y.shape) . astype(NP_FLOAT)
      else:
        print ( "[WARNING] No reweighting strategy available, since there aren't weights to reweight" )

    self._rw_enabled = enable_reweights

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
  
  def extract_model ( self , 
                      player = "gen" , 
                      fine_tuned_layers = None , 
                      freeze_layers = False ) -> list:   # TODO add docstring
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
        raise TypeError ( f"The number of layers to fine-tune should be an integer," 
                          f" instead {type(fine_tuned_layers)} passed." )
    else:
      fine_tuned_layers = num_layers

    layer_list = list()
    for i, layer in enumerate ( model.layers[:-1] ):
      layer._name = f"loaded_{layer.name}"
      if i < (num_layers - fine_tuned_layers): 
        layer.trainable = False
      else:
        layer.trainable = True
      layer_list . append (layer)

    if freeze_layers:
      for layer in layer_list:
        layer.trainable = False

    return layer_list

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
    report.add_markdown ("---")
    report.add_markdown ('<h2 align="center">Training history</h2>')

    n_epochs = len (history.history["mse"])

    ## Learning curves plots (train-set)
    plt.figure (figsize = (8,5), dpi = 100)
    plt.title  ("Learning curves (training set)", fontsize = 14)   # TODO plot loss variance
    plt.xlabel ("Training epochs", fontsize = 12)
    plt.ylabel (f"{self.model.loss_name}", fontsize = 12)
    plt.plot (history.history["d_loss"], linewidth = 1.5, color = "dodgerblue", label = "discriminator")
    plt.plot (history.history["g_loss"], linewidth = 1.5, color = "coral", label = "generator")
    plt.legend (title = "Adversarial players:", loc = "upper right", fontsize = 10)
    y_bottom = min ( min(history.history["d_loss"][int(n_epochs/10):]), min(history.history["g_loss"][int(n_epochs/10):]) )
    y_top    = max ( max(history.history["d_loss"][int(n_epochs/10):]), max(history.history["g_loss"][int(n_epochs/10):]) )
    y_bottom -= 0.2 * np.abs (y_top)
    y_top    += 0.2 * np.abs (y_top)
    plt.ylim (bottom = y_bottom, top = y_top)

    report.add_figure(); plt.clf(); plt.close()

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
    y_bottom -= 0.2 * np.abs (y_top)
    y_top    += 0.2 * np.abs (y_top)
    plt.ylim (bottom = y_bottom, top = y_top)

    report.add_figure(); plt.clf(); plt.close()

    ## Learning curves plots (val-set)
    plt.figure (figsize = (8,5), dpi = 100)
    plt.title  ("Learning curves (validation set)", fontsize = 14)   # TODO plot loss variance
    plt.xlabel ("Training epochs", fontsize = 12)
    plt.ylabel (f"{self.model.loss_name}", fontsize = 12)
    if self._validation_split != 0.0:
      plt.plot (history.history["val_d_loss"], linewidth = 1.5, color = "dodgerblue", label = "discriminator")
      plt.plot (history.history["val_g_loss"], linewidth = 1.5, color = "coral", label = "generator")
      plt.legend (title = "Adversarial players:", loc = "upper right", fontsize = 10)
      y_bottom = min ( min(history.history["val_d_loss"][int(n_epochs/10):]), min(history.history["val_g_loss"][int(n_epochs/10):]) )
      y_top    = max ( max(history.history["val_d_loss"][int(n_epochs/10):]), max(history.history["val_g_loss"][int(n_epochs/10):]) )
      y_bottom -= 0.2 * np.abs (y_top)
      y_top    += 0.2 * np.abs (y_top)
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
    plt.legend (title = "Adversarial players:", loc = "lower left", fontsize = 10)

    report.add_figure(); plt.clf(); plt.close()

    report.add_markdown ("---")

    ## Validation plots
    Y_ref = self.Y
    Y_gen = self._scaler_Y . inverse_transform ( self.generate (self.X_scaled) )

    for i, y_var in enumerate (self.Y_vars):
      report.add_markdown (f'<h2 align="center">{y_var}</h2>')
      self._grid_val_plot ( report, Y_ref[:,i], Y_gen[:,i], x_label = y_var, log_scale = False )
      self._grid_val_plot ( report, Y_ref[:,i], Y_gen[:,i], x_label = y_var, log_scale = True  )
      self._1d_corr_plot  ( report, Y_ref[:,i], Y_gen[:,i], corr_var = "p", x_label = y_var, log_scale = False )
      self._1d_corr_plot  ( report, Y_ref[:,i], Y_gen[:,i], corr_var = "eta", x_label = y_var, log_scale = False )
      self._1d_corr_plot  ( report, Y_ref[:,i], Y_gen[:,i], corr_var = "p", x_label = y_var, log_scale = True )
      self._1d_corr_plot  ( report, Y_ref[:,i], Y_gen[:,i], corr_var = "eta", x_label = y_var, log_scale = True )
      report.add_markdown ("---")
    
  def _grid_val_plot ( self , 
                       report , 
                       x_ref  , 
                       x_gen  ,
                       x_label = None ,
                       log_scale = False ) -> None:
    fig = plt.figure ( figsize = (28, 6), dpi = 100 )
    gs = gridspec.GridSpec ( nrows = 2 , 
                             ncols = 5 ,
                             wspace = 0.25 ,
                             hspace = 0.25 ,
                             width_ratios  = [2, 2, 1, 1, 1] , 
                             height_ratios = [1, 1] )

    ax0 = fig.add_subplot ( gs[0:,0] )
    ax0 . set_xlabel (x_label, fontsize = 12)
    ax0 . set_ylabel ("Candidates", fontsize = 12)
    if self.w_var is not None:
      ref_label = "Original (sWeighted)"
      gen_label = "Generated (reweighted)" if self._rw_enabled else "Generated (sWeighted)"
    else:
      ref_label = "Original (no sWeights)"
      gen_label = "Generated (no sWeights)"
    h_ref, bins, _ = ax0 . hist ( x_ref, bins = 100, weights = self._w_Y, 
                                  density = True, color = "dodgerblue", label = ref_label )
    h_gen, _ , _ = ax0 . hist ( x_gen, bins = bins, weights = self._w_X, histtype = "step", 
                                density = True, color = "deeppink", label = gen_label )
    ax0 . legend (loc = "upper left", fontsize = 10)
    y_max = max ( h_ref.max(), h_gen.max() )
    if log_scale:
      y_min = min ( h_ref[h_ref>0].min(), h_gen[h_gen>0].min() )
      y_max *= 20
      ax0 . set_yscale ("log")
    else:
      y_min = 0.0
      y_max += 0.2 * y_max
    ax0 . set_ylim (bottom = y_min, top = y_max)

    ax1 = fig.add_subplot ( gs[0:,1] )
    ax1 . set_xlabel (x_label, fontsize = 12)
    ax1 . set_ylabel ("Candidates", fontsize = 12)
    ref_label = "Original (sWeighted)" if self.w_var else "Original (no sWeights)"
    gen_label = "Generated"
    h_ref, bins, _ = ax1 . hist ( x_ref, bins = 100, weights = self._w_Y, 
                                  density = True, color = "dodgerblue", label = ref_label )
    h_gen, _ , _ = ax1 . hist ( x_gen, bins = bins, histtype = "step", 
                                density = True, color = "deeppink", label = gen_label )
    ax1 . legend (loc = "upper left", fontsize = 10)
    y_max = max ( h_ref.max(), h_gen.max() )
    if log_scale:
      y_min = min ( h_ref[h_ref>0].min(), h_gen[h_gen>0].min() )
      y_max *= 20
      ax1 . set_yscale ("log")
    else:
      y_min = 0.0
      y_max += 0.2 * y_max
    ax1 . set_ylim (bottom = y_min, top = y_max)

    self._2d_corr_plot ( figure  = fig ,
                         gs_list = [ gs[0,2], gs[1,2] ] ,
                         x_ref = x_ref , 
                         x_gen = x_gen , 
                         y = self.X[:,0]/1e3 ,
                         bins = 25 , 
                         density = True , 
                         w_ref = self._w_Y.flatten() ,
                         w_gen = self._w_X.flatten() ,
                         xlabel = x_label ,
                         ylabel = "Momentum [Gev/$c$]" )

    self._2d_corr_plot ( figure  = fig ,
                         gs_list = [ gs[0,3], gs[1,3] ] ,
                         x_ref = x_ref , 
                         x_gen = x_gen , 
                         y = self.X[:,1] ,
                         bins = 25 , 
                         density = True , 
                         w_ref = self._w_Y.flatten() ,
                         w_gen = self._w_X.flatten() ,
                         xlabel = x_label ,
                         ylabel = "Pseudorapidity" )

    self._2d_corr_plot ( figure  = fig ,
                         gs_list = [ gs[0,4], gs[1,4] ] ,
                         x_ref = x_ref , 
                         x_gen = x_gen , 
                         y = self.X[:,2] ,
                         bins = 25 , 
                         density = True , 
                         w_ref = self._w_Y.flatten() ,
                         w_gen = self._w_X.flatten() ,
                         xlabel = x_label ,
                         ylabel = "$\mathtt{nTracks}$" )

    report.add_figure(options = "width=100%"); plt.clf(); plt.close()
    # report.add_markdown ("<br/>")

  def _2d_corr_plot ( self , 
                      figure  ,
                      gs_list , 
                      x_ref , x_gen , y , 
                      bins = 10 , 
                      density = False , 
                      w_ref   = None  ,
                      w_gen   = None  ,
                      xlabel  = None  ,
                      ylabel  = None  ) -> None:
    """Internal function"""
    if len(gs_list) != 2: raise ValueError ("It should be passed only 2 GridSpec positions.")

    ## Binning definition
    x_min = min ( x_ref.min() , x_gen.min() )
    x_max = max ( x_ref.max() , x_gen.max() )
    x_min -= 0.1 * ( x_max - x_min )
    x_max += 0.1 * ( x_max - x_min )
    y_min = y.min() - 0.1 * ( y.max() - y.min() )
    y_max = y.max() + 0.1 * ( y.max() - y.min() )
    binning = [ np.linspace ( x_min, x_max, bins + 1 ) ,
                np.linspace ( y_min, y_max, bins + 1 ) ]

    ax0 = figure.add_subplot ( gs_list[0] )
    if xlabel: ax0 . set_xlabel ( xlabel, fontsize = 10 )
    if ylabel: ax0 . set_ylabel ( ylabel, fontsize = 10 )
    hist2d = np.histogram2d ( x_ref, y, weights = w_ref, density = density, bins = binning )
    ax0 . pcolormesh ( binning[0], binning[1], hist2d[0].T, cmap = plt.get_cmap ("viridis"), vmin = 0 )
    ax0 . annotate ( "original", color = "w", weight = "bold",
                     ha = "center", va = "center", size = 10,
                     xy = (0.8, 0.9), xycoords = "axes fraction", 
                     bbox = dict (boxstyle = "round", fc = "dodgerblue", alpha = 1.0, ec = "1.0") )

    ax1 = figure.add_subplot ( gs_list[1] )
    if xlabel: ax1 . set_xlabel ( xlabel, fontsize = 10 )
    if ylabel: ax1 . set_ylabel ( ylabel, fontsize = 10 )
    hist2d = np.histogram2d ( x_gen, y, weights = w_gen, density = density, bins = binning )
    ax1 . pcolormesh ( binning[0], binning[1], hist2d[0].T, cmap = plt.get_cmap ("viridis"), vmin = 0 )
    ax1 . annotate ( "generated", color = "w", weight = "bold",
                     ha = "center", va = "center", size = 10,
                     xy = (0.8, 0.9), xycoords = "axes fraction", 
                     bbox = dict (boxstyle = "round", fc = "deeppink", alpha = 1.0, ec = "1.0") )

  def _1d_corr_plot ( self , 
                      report ,
                      x_ref  , 
                      x_gen  ,
                      corr_var = "p" ,
                      x_label = None ,
                      log_scale = False ) -> None:
    """Internal function"""
    if corr_var not in ["p", "eta"]:
      raise ValueError ("error")   # TODO add error message

    fig, ax = plt.subplots ( nrows = 2, ncols = 2, figsize = (14,8), dpi = 100 )
    plt.subplots_adjust ( wspace = 0.25, hspace = 0.25 )

    if self.w_var is not None:
      ref_label = "Original (sWeighted)"
      gen_label = "Generated (reweighted)" if self._rw_enabled else "Generated (sWeighted)"
    else:
      ref_label = "Original (no sWeights)"
      gen_label = "Generated (no sWeights)"

    if corr_var == "p":
      cond = self._X[:,0]
      bounds = [0.1e3, 5e3, 10e3, 25e3, 100e3]
    else:
      cond = self._X[:,1]
      bounds = [1.8, 2.7, 3.5, 4.2, 5.5]

    idx = 0
    for i in range(2):
      for j in range(2):
        ax[i,j] . set_xlabel (x_label, fontsize = 12)
        ax[i,j] . set_ylabel ("Candidates", fontsize = 12)

        query = ( cond >= bounds[idx] ) & ( cond < bounds[idx+1] )
        h_ref, bins, _ = ax[i,j] . hist ( x_ref[query], bins = 100, weights = self._w_Y[query], 
                                          density = True, color = "dodgerblue", label = ref_label )
        h_gen, _ , _ = ax[i,j] . hist ( x_gen[query], bins = bins, weights = self._w_X[query], histtype = "step", 
                                        density = True, color = "deeppink", label = gen_label )

        if corr_var == "p":
          text = f"$p \in ({bounds[idx]/1e3:.1f}, {bounds[idx+1]/1e3:.1f})$ [GeV/$c$]"
        else:
          text = f"$\eta \in ({bounds[idx]:.1f}, {bounds[idx+1]:.1f})$"

        ax[i,j] . annotate ( text, fontsize = 10, ha = "right", va = "top",
                             xy = (0.95, 0.95), xycoords = "axes fraction" )
        ax[i,j] . legend ( loc = "upper left", fontsize = 10 )

        y_max = max ( h_ref.max(), h_gen.max() )
        if log_scale:
          y_min = min ( h_ref[h_ref>0].min(), h_gen[h_gen>0].min() )
          y_max *= 20
          ax[i,j] . set_yscale ("log")
        else:
          y_min = 0.0
          y_max += 0.2 * y_max
        ax[i,j] . set_ylim (bottom = y_min, top = y_max)

        idx += 1

    report.add_figure(options = "width=45%"); plt.clf(); plt.close()

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
