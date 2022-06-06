#from __future__ import annotations

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from html_reports import Report
from matplotlib.patches import Patch
from sklearn.pipeline import Pipeline
from lb_pidsim_train.trainers import BaseTrainer
from lb_pidsim_train.metrics import KL_div_from_counts, JS_div_from_counts, KS_test_from_counts, chi2_test_from_counts


NP_FLOAT = np.float32
"""Default data-type for arrays."""


class ScikitTrainer (BaseTrainer):
  def __init__ ( self ,
                 name ,
                 export_dir  = None ,
                 export_name = None ,
                 report_dir  = None ,
                 report_name = None ,
                 verbose = False ) -> None:   # TODO new variable name for warnings
    super().__init__ ( name = name ,
                       export_dir  = export_dir  ,
                       export_name = export_name ,
                       report_dir  = report_dir  ,
                       report_name = report_name ,
                       verbose = verbose )

  def train_model ( self , 
                    model ,
                    validation_split = 0.2 ,
                    performance_metric = "chi2_test" ,
                    verbose = 0 ) -> None:   # TODO add docstring
    """"""
    if not self._dataset_prepared:
      raise RuntimeError ("error")   # TODO implement error message

    ## Data-type control
    try:
      validation_split = float ( validation_split )
    except:
      raise TypeError ( f"The fraction of train-set used for validation should"
                        f" be a float, instead {type(validation_split)} passed." )

    ## Data-value control
    if (validation_split < 0.0) or (validation_split > 1.0):
      raise ValueError ("error")   # TODO add error message

    if performance_metric not in ["kl_div", "js_div", "ks_test", "chi2_test"]:
      raise ValueError ("error")   # TODO add error message

    self._validation_split   = self._params.get ( "validation_split"   , validation_split   )
    self._performance_metric = self._params.get ( "performance_metric" , performance_metric )

    ## Sizes computation
    sample_size = self._X . shape[0]
    trainset_size = int ( (1.0 - validation_split) * sample_size )

    ## Training dataset
    train_feats  = self._X_scaled[:trainset_size]
    train_labels = self._Y[:trainset_size] . flatten()
    train_w      = self._w[:trainset_size] . flatten()

    ## Training procedure
    start = datetime.now()
    model . fit (train_feats, train_labels, sample_weight = train_w)
    self._model_trained = True   # switch on model trained flag
    stop  = datetime.now()
    if (verbose > 0): 
      timestamp = str(stop-start) . split (".") [0]   # HH:MM:SS
      timestamp = timestamp . split (":")   # [HH, MM, SS]
      timestamp = f"{timestamp[0]}h {timestamp[1]}min {timestamp[2]}s"
      print ( f"[INFO] Classifier training completed in {timestamp}" )

    self._model = model
    self._save_model ( "saved_model", model, verbose = (verbose > 0) )
    self._save_pipeline ( verbose = (verbose > 0) )

    self._scores = [None, None]   # score init

    ## Report setup
    report = Report()   # TODO add hyperparams to the report
    self._report_params (report)
    if validation_split != 0.0:
      report.add_markdown ("---")
      report.add_markdown ('<h2 align="center">Model performance on validation set</h2>')
      self._eff_hist2d (report, bins = 100, validation = True)
      self._eff_hist1d (report, bins = 25, validation = True)
      self._report_score (report, validation = True)
    report.add_markdown ("---")
    report.add_markdown ('<h2 align="center">Model performance on training set</h2>')
    self._eff_hist2d (report, bins = 100, validation = False)
    self._eff_hist1d (report, bins = 25, validation = False)
    self._report_score (report, validation = False)

    filename = f"{self._report_dir}/{self._report_name}"
    report . write_report ( filename = f"{filename}.html" )
    if (verbose > 0):
      print ( f"[INFO] Training report correctly exported to {filename}" )

  def _report_params (self, report) -> None:
    report.add_markdown ("---")
    report.add_markdown ('<h2 align="center">Hyperparameters and other details</h2>')
    params_dict = self._params.get_dict()
    text = ""
    for k in params_dict.keys():
      text += f"**{k}** : {params_dict[k]}  \n"
    report.add_markdown (text)

  # +------------------------+
  # |    Validation plots    |
  # +------------------------+

  def _eff_hist2d ( self, report, bins = 100, validation = False ) -> None:   # TODO add docstring
    """"""
    X, Y, w, probas = self._data_to_plot ( validation = validation )

    binning = [ np.linspace ( 0 , 100 , bins+1 ) ,   # momentum binning
                np.linspace ( 1 , 6   , bins+1 ) ]   # pseudorapidity binning

    ## Efficiency correction
    plt.figure (figsize = (8,5), dpi = 100)
    plt.title  ("isMuon (Data sample)", fontsize = 14)
    plt.xlabel ("Momentum [GeV/$c$]", fontsize = 12)
    plt.ylabel ("Pseudorapidity", fontsize = 12)
    hist2d = np.histogram2d ( X[:,0][Y == 1]/1e3, X[:,1][Y == 1], weights = w[Y == 1], bins = binning )
    plt.pcolormesh ( binning[0], binning[1], hist2d[0].T, cmap = plt.get_cmap ("viridis"), vmin = 0 )

    report.add_figure (options = "width=45%"); plt.clf(); plt.close()

    ## Efficiency parameterization
    plt.figure (figsize = (8,5), dpi = 100)
    plt.title  (f"isMuon (Trained model)", fontsize = 14)
    plt.xlabel ("Momentum [GeV/$c$]", fontsize = 12)
    plt.ylabel ("Pseudorapidity", fontsize = 12)
    hist2d = np.histogram2d ( X[:,0]/1e3, X[:,1], weights = w * probas, bins = binning )
    plt.pcolormesh ( binning[0], binning[1], hist2d[0].T, cmap = plt.get_cmap ("viridis"), vmin = 0 )

    report.add_figure (options = "width=45%"); plt.clf(); plt.close()

  def _eff_hist1d ( self , 
                    report , 
                    bins = 100 , 
                    validation = False ) -> None:   # TODO add docstring
    """"""
    X, Y, w, probas = self._data_to_plot ( validation = validation )
    eta_limits = [ 1.8, 2.7, 3.5, 4.2, 5.5 ]   ## Pseudorapidity limits
    p_bins = np.linspace ( 0.1, 100.0, bins+1 ) 

    if not validation:
      self._scores[0] = list()
    else:
      self._scores[1] = list()

    for i in range (len(eta_limits) - 1):
      fig, ax = plt.subplots (nrows = 1, ncols = 2, figsize = (16,5), dpi = 100)

      ## Left plot
      ax[0].set_title  (f"isMuon for $\eta$ in ({eta_limits[i]}, {eta_limits[i+1]})")
      ax[0].set_xlabel ("Momentum [GeV/$c$]", fontsize = 12)
      ax[0].set_ylabel ("Entries", fontsize = 12)
      ax[0].set_yscale ("log")
  
      custom_handles = list()
      custom_labels = list()
  
      query = (X[:,1] > eta_limits[i]) & (X[:,1] <= eta_limits[i+1])   # NumPy query

      ## TurboCalib
      h_all, bins_all = np.histogram ( X[:,0][query]/1e3, bins = p_bins, weights = w[query] )
      h_all = np.where ( h_all > 0, h_all, 0 ) . astype (np.int32)   # ensure positive entries

      bin_centers = ( bins_all[1:] + bins_all[:-1] ) / 2
      ax[0].errorbar ( bin_centers, h_all, yerr = 0.0, color = "red", drawstyle = "steps-mid", zorder = 2 )
      custom_handles . append ( Patch (facecolor = "white", alpha = 0.8, edgecolor = "red") )
      custom_labels  . append ( "TurboCalib" )
  
      ## Entries (efficiency parameterization)
      h_pred, _ = np.histogram ( X[:,0][query]/1e3, bins = bins_all, weights = w[query] * probas[query] )
      h_pred = np.where ( h_pred > 0, h_pred, 0 ) . astype (np.int32)   # ensure positive entries

      ax[0].errorbar ( bin_centers, h_pred, yerr = 0.0, color = "royalblue", drawstyle = "steps-mid", zorder = 1 )
      custom_handles . append ( Patch (facecolor = "white", alpha = 0.8, edgecolor = "royalblue") )
      custom_labels  . append ( f"isMuon model" )
  
      ## Entries (efficiency correction)
      h_true, _ = np.histogram ( X[:,0][query][Y[query] == 1]/1e3, bins = bins_all, weights = w[query][Y[query] == 1] )
      h_true = np.where ( h_true > 0, h_true, 0 ) . astype (np.int32)   # ensure positive entries

      ax[0].errorbar ( bin_centers, h_true, yerr = h_true**0.5, fmt = '.', color = "black", 
                       barsabove = True, capsize = 2, label = f"isMuon passed", zorder = 0 )
      handles, labels = ax[0].get_legend_handles_labels()
      custom_handles . append ( handles[-1] )
      custom_labels  . append ( labels[-1] )
  
      ax[0].legend (handles = custom_handles, labels = custom_labels, loc = "upper right", fontsize = 10)

      ## Score computation
      score = self._compute_score ( h_pred, h_true, strategy = self._performance_metric )
      if not validation:
        self._scores[0] . append ( score )
      else:
        self._scores[1] . append ( score )

      ## Right plot
      ax[1].set_title  (f"isMuon for $\eta$ in ({eta_limits[i]}, {eta_limits[i+1]})")
      ax[1].set_xlabel ("Momentum [GeV/$c$]", fontsize = 12)
      ax[1].set_ylabel ("isMuon efficiency", fontsize = 12)

      ## Efficiency parameterization
      eff_pred = 99. * np.ones_like(h_true) . astype (np.float32)
      eff_pred[h_all>0] = h_pred[h_all>0] / h_all[h_all>0]
      eff_pred[eff_pred>1] = 1   # constrain efficiency to 1.0

      h_all_err = np.zeros_like(h_all) . astype (np.float32)
      h_all_err[h_all>0] = np.sqrt(h_all[h_all>0]) / h_all[h_all>0]

      h_pred_err = np.zeros_like(h_pred) . astype (np.float32)
      h_pred_err[h_all>0] = np.sqrt(h_pred[h_all>0]) / h_pred[h_all>0]
      h_pred_err[h_pred==0] = 0.0
      
      eff_pred_err = np.zeros_like(eff_pred) . astype (np.float32)
      eff_pred_err[h_all>0] = eff_pred[h_all>0] * np.sqrt( h_pred_err[h_all>0]**2 + h_all_err[h_all>0]**2 )
      
      ## Efficiency correction
      eff_true = 99. * np.ones_like(h_true) . astype (np.float32)
      eff_true[h_all>0] = h_true[h_all>0] / h_all[h_all>0]
      eff_true[eff_true>1] = 1   # constrain efficiency to 1.0

      h_true_err = np.zeros_like(h_true) . astype (np.float32)
      h_true_err[h_all>0] = np.sqrt(h_true[h_all>0]) / h_true[h_all>0]
      h_true_err[h_true==0] = 0.0

      eff_true_err = np.zeros_like(eff_true) . astype (np.float32)
      eff_true_err[h_all>0] = eff_true[h_all>0] * np.sqrt( h_true_err[h_all>0]**2 + h_all_err[h_all>0]**2 )

      ax[1].errorbar ( bin_centers, eff_true, yerr = eff_true_err, marker = "o", markersize = 5, capsize = 3, elinewidth = 2, 
                       mec = "coral", mfc = "w", color = "coral", label = f"Data sample", zorder = 0 )
      ax[1].errorbar ( bin_centers, eff_pred, yerr = eff_pred_err, marker = "o", markersize = 5, capsize = 3, elinewidth = 1, 
                       mec = "forestgreen", mfc = "w", color = "forestgreen", label = f"Trained model", zorder = 1 )

      ax[1].legend (loc = "upper right", fontsize = 10)

      report.add_figure (options = "width=100%"); plt.clf(); plt.close()

  def _report_score (self, report, validation = False) -> None:
    """Internal method."""
    if   ( self._performance_metric == "kl_div"    ) : perf_key = "K-L div"
    elif ( self._performance_metric == "js_div"    ) : perf_key = "J-S div"
    elif ( self._performance_metric == "ks_test"   ) : perf_key = "K-S test"
    elif ( self._performance_metric == "chi2_test" ) : perf_key = "chi2 test"
    perf_score = np.max ( self._scores[1] ) if validation else np.max ( self._scores[0] )
    report.add_markdown ( f"**{perf_key}** : {perf_score:.2e}" )

  def _data_to_plot (self, validation = False) -> tuple:   # TODO complete docstring
    """...
    
    Parameters
    ----------  
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

    probas : `np.ndarray`
      ...
    """
    sample_size = self._X . shape[0]
    trainset_size = int ( (1.0 - self._validation_split) * sample_size )
    if not validation:
      X, X_scaled = self._X[:trainset_size], self._X_scaled[:trainset_size]
      Y, w = self._Y[:trainset_size], self._w[:trainset_size]
    else:
      if self._validation_split == 0.0:
        raise ValueError ("error.")   # TODO add error message
      X, X_scaled = self._X[trainset_size:], self._X_scaled[trainset_size:]
      Y, w = self._Y[trainset_size:], self._w[trainset_size:]
    probas = self.model.predict_proba ( X_scaled ) [:,1]
    return X, Y.flatten(), w.flatten(), probas.flatten()

  def _compute_score ( self ,
                       h_pred ,
                       h_true ,
                       strategy = "ks_test" ) -> float:   # TODO add docstring
    """short description"""
    if strategy == "chi2_test":
      score = chi2_test_from_counts (h_pred, h_true)
    elif strategy == "ks_test":
      score = KS_test_from_counts (h_pred, h_true)
    elif strategy == "kl_div":
      score = KL_div_from_counts (h_pred, h_true)
    elif strategy == "js_div":
      score = JS_div_from_counts (h_pred, h_true)
    else:
      ValueError ("error.")   # TODO add error message
    return score

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
    if verbose: print ( f"[INFO] Trained model correctly exported to {filename}" )

  def _save_pipeline ( self, verbose = False ) -> None:   # TODO complete docstring
    """"""
    t_models = list()
    t_names  = ["transform_X", "transform_Y", "saved_model"]
    dirname  = f"{self._export_dir}/{self._export_name}"
    for n in t_names:
      filename = f"{dirname}/{n}.pkl"
      if os.path.exists (filename):
        with open (filename, "rb") as f:
          transformer = pickle.load (f)
        t_models . append ( (n, transformer) )
    pipeline = Pipeline ( t_models )
    pickle . dump ( pipeline, open (f"{dirname}/pipeline.pkl", "wb") )
    if verbose: print ( f"[INFO] Pipeline correctly exported to {dirname}/pipeline.pkl" )

  @property
  def model (self):
    """The model after the training procedure."""
    return self._model

  @property
  def scores (self) -> list:
    """Model quality scores on training and validation sets."""
    return self._scores
