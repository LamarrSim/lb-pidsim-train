from __future__ import annotations

import os
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

from lb_pidsim_train.algorithms.gan import GAN


d_loss_tracker = tf.keras.metrics.Mean ( name = "d_loss" )
"""Metric instance to track the discriminator loss score."""

g_loss_tracker = tf.keras.metrics.Mean ( name = "g_loss" )
"""Metric instance to track the generator loss score."""


class BceGAN (GAN):
  """Keras model class to build and train BceGAN system.
  
  Parameters
  ----------
  discriminator : `tf.keras.Sequential`
    ...

  generator : `tf.keras.Sequential`
    ...

  latent_dim : `int`, optional
    ... (`64`, by default).

  Attributes
  ----------
  discriminator : `tf.keras.Sequential`
    ...

  generator : `tf.keras.Sequential`
    ...

  latent_dim : `int`
    ...

  Notes
  -----
  ...

  References
  ----------
  ...

  See Also
  --------
  ...

  Methods
  -------
  ...
  """
  def _compute_g_loss (self, gen_sample, ref_sample, weights = None) -> tf.Tensor:
    """Return the generator loss.
    
    Parameters
    ----------
    gen_sample : `tf.Tensor`
      ...

    ref_sample : `tf.Tensor`
      ...

    weights : `tf.Tensor`, optional
      ... (`None`, by default).

    Returns
    -------
    g_loss : `tf.Tensor`
      ...
    """
    ## Noise injection to stabilize BceGAN training
    rnd_gen = tf.random.normal ( tf.shape(gen_sample), mean = 0., stddev = 0.1 )
    rnd_ref = tf.random.normal ( tf.shape(ref_sample), mean = 0., stddev = 0.1 )
    D_gen = self._discriminator ( gen_sample + rnd_gen )
    D_ref = self._discriminator ( ref_sample + rnd_ref )

    ## Loss computation
    true_gen = 0.9
    true_ref = 0.1
    loss = true_gen       * tf.math.log ( tf.clip_by_value ( D_gen     , 1e-12 , 1. ) ) + \
           (1 - true_gen) * tf.math.log ( tf.clip_by_value ( 1 - D_gen , 1e-12 , 1. ) ) + \
           true_ref       * tf.math.log ( tf.clip_by_value ( D_ref     , 1e-12 , 1. ) ) + \
           (1 - true_ref) * tf.math.log ( tf.clip_by_value ( 1 - D_ref , 1e-12 , 1. ) ) 
    if weights is not None:
      loss = weights * loss
    return tf.reduce_mean (loss)
    