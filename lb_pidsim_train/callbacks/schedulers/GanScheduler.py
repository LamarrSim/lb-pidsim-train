#from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.callbacks import Callback

K = tf.keras.backend

class GanScheduler (Callback):   # TODO add docstring
  """class description"""
  def __init__(self):   # TODO add data-type check
    super(GanScheduler, self) . __init__()

  def on_epoch_begin (self, epoch, logs = None):
    if epoch != 0:
      scale_factor = self._compute_scale_factor (epoch = epoch)
    else:
      scale_factor = 1.0

    ## Discriminator lr-scheduling
    d_lr0 = K.get_value ( self.model.d_lr0 )
    K.set_value ( self.model.d_optimizer.learning_rate, d_lr0 * scale_factor )

    ## Generator lr-scheduling
    g_lr0 = K.get_value ( self.model.g_lr0 )
    K.set_value ( self.model.g_optimizer.learning_rate, g_lr0 * scale_factor )

  def _compute_scale_factor (self, epoch):
    return 1.0

  def on_epoch_end (self, epoch, logs = None):
    logs = logs or {}
    logs["d_lr"] = K.get_value ( self.model.d_optimizer.learning_rate )
    logs["g_lr"] = K.get_value ( self.model.g_optimizer.learning_rate )
