#from __future__ import annotations

import tensorflow as tf

from lb_pidsim_train.trainers import BaseTrainer

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


TF_FLOAT = tf.float32
"""Default data-type for tensors."""

NP_FLOAT = TF_FLOAT.as_numpy_dtype
"""Default data-type for arrays."""


class GanTrainer (BaseTrainer):
  def feed_from_root_files ( self , 
                             root_files , 
                             X_vars = None , 
                             Y_vars = None , 
                             w_var  = None , 
                             selections = None , 
                             tree_names = None , 
                             chunk_size = None ,
                             verbose = 0 ) -> None:
    """Feed the training procedure with ROOT files.
    
    Parameters
    ----------
    root_files : `str` or `list` of `str`
      List of ROOT files used for the training procedure.

    X_vars : `str` or `list` of `str`, optional
      Branch names of the input variables within the ROOT trees 
      (`None`, by default).

    Y_vars : `str` or `list` of `str`, optional
      Branch names of the output variables within the ROOT trees 
      (`None`, by default).
    
    w_var : `str` or `list` of `str`, optional
      Branch name of the weight variable, if available, within the 
      ROOT trees (`None`, by default).

    selections : `str` or `list` of `str`, optional
      Boolean expressions to filter the ROOT trees (`None`, by default).

    tree_names : `str` or `list` of `str`, optional
      If more than one ROOT tree is defined for each file, the ones to 
      be loaded have to be defined specifying their names as the keys 
      (`None`, by default).

    chunk_size : `int` or `list` of `int`, optional
      Total number of instance rows loaded to disk as `tf.data.Dataset`
      enabling to handle large amount of data and perform complex 
      transformations (`None`, by default).

    verbose : {0, 1}, optional
      Verbosity mode. `0` = silent (default), `1` = time for data-chunk 
      loading is printed. 

    See Also
    --------
    lb_pidsim_train.utils.data_from_trees :
      Stratified data shuffling from list of `uproot.TTree`.

    tf.data.Dataset :
      Abstraction over a data pipeline that can pull data from several 
      sources, as well as efficiently apply various data transformations.
    """
    super(GanTrainer, self) . feed_from_root_files ( root_files = root_files , 
                                                     X_vars = X_vars , 
                                                     Y_vars = Y_vars , 
                                                     w_var  = w_var  , 
                                                     selections = selections , 
                                                     tree_names = tree_names , 
                                                     chunk_size = chunk_size ,
                                                     verbose = verbose )

  def train_model ( self ,
                    model ,
                    batch_size ,
                    num_epochs ) -> None:
    """short description"""
    dataset = self._create_dataset (batch_size=batch_size)
    history = model . fit ( dataset, batch_size = batch_size, epochs = num_epochs, verbose = 1 )
    return history

  def _create_dataset ( self, buffer_size = 100, batch_size = 100 ) -> tf.data.Dataset:
    """short description"""
    X = tf.cast ( tf.convert_to_tensor(self._X), dtype = TF_FLOAT )
    Y = tf.cast ( tf.convert_to_tensor(self._Y), dtype = TF_FLOAT )
    w = tf.cast ( tf.convert_to_tensor(self._w), dtype = TF_FLOAT )

    dataset = tf.data.Dataset.from_tensor_slices ( (X, Y, w) )
    dataset = dataset.shuffle (buffer_size)
    dataset = dataset.batch (batch_size)
    return dataset