#from __future__ import annotations

import os
import pickle
import numpy as np
import tensorflow as tf

from warnings import warn
from sklearn.utils import shuffle
from lb_pidsim_train.trainers import BaseTrainer
from lb_pidsim_train.utils import warn_message as wm


NP_FLOAT = np.float32
"""Default data-type for arrays."""

TF_FLOAT = tf.float32
"""Default data-type for tensors."""


class BdtClassifier (BaseTrainer):
  def __init__ ( self , 
                 name ,
                 model_dir  ,
                 model_name , 
                 export_dir  = None , 
                 export_name = None , 
                 report_dir  = None , 
                 report_name = None , 
                 verbose = False ) -> None:

    self._generator = tf.keras.models.load_model (f"{model_dir}/{model_name}/saved_model")

    if export_dir is None:
      export_dir = "./models"
      message = wm.name_not_passed ("export dirname", export_dir)
      if verbose: warn (message)
    self._export_dir = export_dir
    if not os.path.exists (self._export_dir):
      message = wm.directory_not_found (self._export_dir)
      if verbose: warn (message)
      os.makedirs (self._export_dir)

    if export_name is None:
      export_name = f"{name}_{model_name}"
      message = wm.name_not_passed ("export filename", export_name)
      if verbose: warn (message)
    self._export_name = export_name

    if report_dir is None:
      report_dir = "./reports"
      message = wm.name_not_passed ("report dirname", report_dir)
      if verbose: warn (message)
    self._report_dir = report_dir
    if not os.path.exists (self._report_dir):
      message = wm.directory_not_found (self._report_dir)
      if verbose: warn (message)
      os.makedirs (self._report_dir)

    if report_name is None:
      report_name = f"{name}_{model_name}"
      message = wm.name_not_passed ("report filename", report_name)
      if verbose: warn (message)
    self._report_name = report_name

  def prepare_dataset(self, X_preprocessing=None, Y_preprocessing=None, X_vars_to_preprocess=None, Y_vars_to_preprocess=None, subsample_size=100000, save_transformer=True, verbose=0) -> None:
    return super().prepare_dataset(X_preprocessing=X_preprocessing, Y_preprocessing=Y_preprocessing, X_vars_to_preprocess=X_vars_to_preprocess, Y_vars_to_preprocess=Y_vars_to_preprocess, subsample_size=subsample_size, save_transformer=save_transformer, verbose=verbose)

  def train_model ( self ,
                    model ,
                    validation_split = 0.0 ,
                    plots_on_report = True ,
                    save_model = True ,
                    verbose = 0 ) -> None:
    ## Data-type control
    try:
      validation_split = float ( validation_split )
    except:
      raise TypeError ( f"The fraction of train-set used for validation should"
                        f" be a float, instead {type(validation_split)} passed." )

    ## Data-value control
    if (validation_split < 0.0) or (validation_split > 1.0):
      raise ValueError ("error")   # docs to add

    self._validation_split = validation_split

    ## Sizes computation
    sample_size = self._X . shape[0]
    trainset_size = int ( (1.0 - validation_split) * sample_size )

    ## Training dataset
    trainset = ( self._X_scaled[:trainset_size], self._Y_scaled[:trainset_size], self._w[:trainset_size] )
    feats, labels, w = self._create_dataset ( data = trainset, model = self._generator, latent_dim = 64)

    print (self._w)
    print (w)

    model . fit (feats, labels, sample_weight = w)
    print (model.predict_proba(feats[0,:]))

  @staticmethod
  def _create_dataset (data, model, latent_dim) -> tuple:
    batch_size = int ( data[0].shape[0] / 2 )

    X_gen = np.array ( data[0][:batch_size], dtype = NP_FLOAT )
    X_ref = np.array ( data[0][batch_size:], dtype = NP_FLOAT )

    X_gen_tensor  = tf.convert_to_tensor (X_gen)
    latent_tensor = tf.random.normal ( shape = (batch_size, latent_dim) )
    input_tensor  = tf.concat ( [X_gen_tensor, latent_tensor], axis = 1 )

    Y_gen = model ( tf.cast (input_tensor, dtype = TF_FLOAT) ) . numpy() # manca il preprocessing
    Y_ref = np.array ( data[1][batch_size:], dtype = NP_FLOAT )

    XY_gen = np.c_ [X_gen, Y_gen]
    XY_ref = np.c_ [X_ref, Y_ref]

    feats = np.r_ [XY_gen, XY_ref]
    labels = np.concatenate ( [ np.ones (len(X_gen)) , np.zeros (len(X_ref)) ] )
    weights = np.array ( data[2], dtype = NP_FLOAT ) . reshape (len(labels),)
    
    feats, labels, weights = shuffle (feats, labels, weights)
    return feats, labels, weights
