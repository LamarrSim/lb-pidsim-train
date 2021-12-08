#from __future__ import annotations

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from html_reports import Report
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score
from lb_pidsim_train.trainers import BaseTrainer


NP_FLOAT = np.float32
"""Default data-type for arrays."""


class ScikitTrainer (BaseTrainer):
  def train_model ( self , 
                    model ,
                    validation_split = 0.2 ,
                    plots_on_report = True ,
                    save_model = True ,
                    verbose = 0 ) -> None:   # TODO add docstring
    """"""
    report = Report()   # TODO add hyperparams to the report

    ## Data-type control
    try:
      validation_split = float ( validation_split )
    except:
      raise TypeError ( f"The fraction of train-set used for validation should"
                        f" be a float, instead {type(validation_split)} passed." )

    ## Data-value control
    if (validation_split < 0.0) or (validation_split > 1.0):
      raise ValueError ("error")   # TODO add error message

    self._validation_split = validation_split

    ## Sizes computation
    sample_size = self._X . shape[0]
    trainset_size = int ( (1.0 - validation_split) * sample_size )

    ## Training dataset
    train_feats  = self._X_scaled[:trainset_size]
    train_labels = self._Y[:trainset_size] . flatten()
    train_w      = self._w[:trainset_size] . flatten()

    ## Validation dataset
    if validation_split != 0.0:
      val_feats  = self._X_scaled[trainset_size:]
      val_labels = self._Y[trainset_size:] . flatten()
      val_w      = self._w[trainset_size:] . flatten()

    ## Training procedure
    start = datetime.now()
    model . fit (train_feats, train_labels, sample_weight = train_w)
    stop  = datetime.now()
    if (verbose > 0): 
      timestamp = str(stop-start) . split (".") [0]   # HH:MM:SS
      timestamp = timestamp . split (":")   # [HH, MM, SS]
      timestamp = f"{timestamp[0]}h {timestamp[1]}min {timestamp[2]}s"
      print (f"Classifier training completed in {timestamp}.")

    self._model = model
    self._scores = [None, None]

    self._scores[0] = roc_auc_score ( train_labels, model.predict_proba(train_feats)[:,1] )
    if validation_split != 0.0:
      self._scores[1] = roc_auc_score ( val_labels, model.predict_proba(val_feats)[:,1] )

    if plots_on_report:
      if validation_split != 0.0:
        report.add_markdown ("## Model performance on validation set")
        self._eff_hist2d (report, bins = 100, validation = True)
        self._eff_hist1d (report, bins = 50, validation = True)
        report.add_markdown ("***")
      report.add_markdown ("## Model performance on training set")
      self._eff_hist2d (report, bins = 100, validation = False)
      self._eff_hist1d (report, bins = 50, validation = False)

    if save_model:
      self._save_model ( f"{self._name}", model, verbose = (verbose > 0) )

    filename = f"{self._report_dir}/{self._report_name}"
    report . write_report ( filename = f"{filename}.html" )
    if (verbose > 1):
      print (f"Training report correctly exported to {filename}")

  def _eff_hist2d ( self, report, bins = 100, validation = False ) -> None:   # TODO add docstring
    """"""
    rng = [ [0, 100] , [1, 6] ]   # momentum-pseudorapidity ranges

    plt.figure (figsize = (8,5), dpi = 100)
    plt.title  ("isMuon (Brunel reconstruction)", fontsize = 14)
    plt.xlabel ("Momentum [GeV/$c$]", fontsize = 12)
    plt.ylabel ("Pseudorapidity", fontsize = 12)
    X, Y, w = self._data_to_plot ( data_from_model = False, validation = validation )
    p = X[:,0][Y == 1] / 1e3   # momentum
    e = X[:,1][Y == 1]   # pseudorapidity
    w = np.where (w[Y == 1] > 0, w[Y == 1], 0.0)   # set negative-weights to zero
    plt.hist2d ( p, e, bins = bins, range = rng, weights = w, density = False, cmap = plt.get_cmap ("inferno") )

    report.add_figure(); plt.clf(); plt.close()

    plt.figure (figsize = (8,5), dpi = 100)
    algo_name = self._name . split("_") [0]
    plt.title  (f"isMuon ({algo_name} model)", fontsize = 14)
    plt.xlabel ("Momentum [GeV/$c$]", fontsize = 12)
    plt.ylabel ("Pseudorapidity", fontsize = 12)
    X, Y, w = self._data_to_plot ( data_from_model = True, validation = validation )
    p = X[:,0][Y == 1] / 1e3   # momentum
    e = X[:,1][Y == 1]   # pseudorapidity
    w = w[Y == 1]   # predict_proba from Scikit-Learn model
    plt.hist2d ( p, e, bins = bins, range = rng, weights = w, density = False, cmap = plt.get_cmap ("inferno") )

    report.add_figure(); plt.clf(); plt.close()
#    report.add_markdown ("<br/>")

  def _eff_hist1d ( self , 
                    report , 
                    bins = 100 , 
                    validation = False , 
                    show_whole_sample = True ) -> None:   # TODO add docstring
    """"""
    model_name = self._name . split("_") [1]
    p_limits = [0, 4, 8, 12, 100]

    for i in range (len(p_limits) - 1):
      binning = [ np.linspace (1, 6, bins+1) ,                     # pseudorapidity binning
                  np.linspace (p_limits[i], p_limits[i+1], bins+1) ]   # momentum binning
  
      fig, ax = plt.subplots (figsize = (8,5), dpi = 100)
      ax.set_title  (f"{model_name} for $p$ in ({p_limits[i]}, {p_limits[i+1]}) GeV/$c$")
      ax.set_xlabel ("Pseudorapidity", fontsize = 12)
      ax.set_ylabel ("Entries", fontsize = 12)
  
      ## Data
      X_true, Y_true, w_true = self._data_to_plot ( data_from_model = False, validation = validation )
      X_pred, Y_pred, w_pred = self._data_to_plot ( data_from_model = True, validation = validation )
  
      custom_handles = list()
      custom_labels = list()
  
      ## TurboCalib
      if show_whole_sample:
        ax.set_yscale ("log")
        ax.hist ( X_true[:,1], bins = binning[0], color = "red", histtype = "step", zorder = 2 )
        custom_handles . append ( Patch (facecolor = "white", alpha = 0.8, edgecolor = "red") )
        custom_labels . append ( "TurboCalib" )
  
      ## Efficiency parameterization
      p_pred = X_pred[:,0][Y_pred == 1] / 1e3   # momentum
      e_pred = X_pred[:,1][Y_pred == 1]   # pseudorapidity
      w_pred = w_pred[Y_pred == 1]   # predict_proba from Scikit-Learn model
  
      entries2d, eta_edges, _ = np.histogram2d ( e_pred, p_pred, bins = binning, weights = w_pred )
      entries = np.sum ( entries2d, axis = 1 )
      eta_centers = ( eta_edges[1:] + eta_edges[:-1] ) / 2
      ax.errorbar ( eta_centers, entries, yerr = 0.0, color = "royalblue", drawstyle = "steps-mid", zorder = 1 )
      custom_handles . append ( Patch (facecolor = "white", alpha = 0.8, edgecolor = "royalblue") )
      custom_labels . append ( f"{model_name} model" )
  
      ## Efficiency correction
      p_true = X_true[:,0][Y_true == 1] / 1e3   # momentum
      e_true = X_true[:,1][Y_true == 1]   # pseudorapidity
      w_true = np.where (w_true[Y_true == 1] > 0, w_true[Y_true == 1], 0.0)   # set negative-weights to zero
  
      entries2d, eta_edges, _ = np.histogram2d ( e_true, p_true, bins = binning, weights = w_true )
      entries = np.sum ( entries2d, axis = 1 )
      eta_centers = ( eta_edges[1:] + eta_edges[:-1] ) / 2
      ax.errorbar ( eta_centers, entries, yerr = entries**0.5, fmt = '.', color = "black", 
                    barsabove = True, capsize = 2, label = f"{model_name} passed", zorder = 0 )
      handles, labels = ax.get_legend_handles_labels()
      custom_handles . append ( handles[-1] )
      custom_labels . append ( labels[-1] )
  
      ax.legend (handles = custom_handles, labels = custom_labels, loc = "upper right", fontsize = 10)
      report.add_figure(); plt.clf(); plt.close()
#    report.add_markdown ("<br/>")

  def _data_to_plot ( self , 
                      data_from_model = False ,
                      validation = False ) -> tuple:   # TODO complete docstring
    """...
    
    Parameters
    ----------
    data_from_model : `bool`
      ...
      
    validation : `bool`
      ...
      
    Returns
    -------
    X : `np.ndarray`
      ...

    Y : `np.ndarray`
      ...

    w : `np.ndarray`
      ...
    """
    sample_size = self._X . shape[0]
    trainset_size = int ( (1.0 - self._validation_split) * sample_size )
    if not validation:
      ## Reference data on train-set
      if not data_from_model:
        X, Y, w = self._X[:trainset_size], self._Y[:trainset_size], self._w[:trainset_size]
      ## Predicted data on train-set
      else:
        X = self._X[:trainset_size]
        Y = self.model.predict ( self._X_scaled[:trainset_size] )
        w = self.model.predict_proba ( self._X_scaled[:trainset_size] ) [:,1]
    else:
      if self._validation_split == 0.0:
        raise ValueError ("error.")   # TODO add error message
      ## Reference data on test-set
      if not data_from_model:
        X, Y, w = self._X[trainset_size:], self._Y[trainset_size:], self._w[trainset_size:]
      ## Predicted data on test-set
      else:
        X = self._X[trainset_size:]
        Y = self.model.predict ( self._X_scaled[trainset_size:] )
        w = self.model.predict_proba ( self._X_scaled[trainset_size:] ) [:,1]
    return X, Y.flatten(), w.flatten()

  def _save_model ( self, name, model, verbose = False ) -> None:   # TODO complete docstring
    """Save the trained model.
    
    Parameters
    ----------
    name : `str`
      Name of the pickle file containing the Scikit-Learn model.

    model : ...
      ...

    verbose : `bool`, optional
      Verbosity mode. `False` = silent (default), `True` = a control message is printed.
    """
    dirname = f"{self._export_dir}/{self._export_name}"
    if not os.path.exists (dirname):
      os.makedirs (dirname)
    filename = f"{dirname}/{name}.pkl"
    pickle . dump ( model, open (filename, "wb") )
    if verbose: print ( f"Trained model correctly exported to {filename}" )

  @property
  def model (self):
    """The model after the training procedure."""
    return self._model

  @property
  def scores (self) -> list:
    """Model quality scores on training and validation sets."""
    return self._scores
