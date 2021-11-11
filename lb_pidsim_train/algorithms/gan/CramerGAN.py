#from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from lb_pidsim_train.algorithms.gan import GAN


d_loss_tracker = tf.keras.metrics.Mean ( name = "d_loss" )
"""Metric instance to track the discriminator loss score."""

g_loss_tracker = tf.keras.metrics.Mean ( name = "g_loss" )
"""Metric instance to track the generator loss score."""


class Critic:
  """Critic function."""
  def __init__ (self, h):
    self.h = h

  def __call__ (self, x_1, x_2):
    critic_func = tf.norm (self.h(x_1) - self.h(x_2), axis = 1) - tf.norm (self.h(x_1), axis = 1)
    return critic_func


class CramerGAN (GAN):
  """Keras model class to build and train CramerGAN system.
  
  Parameters
  ----------
  X_shape : `int` or array_like
    ...

  Y_shape : `int` or array_like
    ...

  discriminator : `list` of `tf.keras.layers`
    ...

  generator : `list` of `tf.keras.layers`
    ...

  latent_dim : `int`, optional
    ... (`64`, by default).

  critic_dim : `int`, optional
    ... (`64`, by default).

  Attributes
  ----------
  discriminator : `tf.keras.Sequential`
    ...

  generator : `tf.keras.Sequential`
    ...

  latent_dim : `int`
    ...

  critic_dim : `int`
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
  def __init__ ( self ,
                 X_shape ,
                 Y_shape ,
                 discriminator ,
                 generator     ,
                 latent_dim = 64 ,
                 critic_dim = 64 ) -> None:
    super(CramerGAN, self) . __init__ ( X_shape = X_shape ,
                                        Y_shape = Y_shape ,
                                        discriminator = discriminator , 
                                        generator     = generator     ,
                                        latent_dim    = latent_dim    )
    self._loss_name = "Energy distance"

    ## Data-type control
    try:
      critic_dim = int ( critic_dim )
    except:
      raise TypeError ("The critic space dimension should be an integer.")

    self._critic_dim = critic_dim

    ## Discriminator sequential model
    self._discriminator = Sequential ( name = "discriminator" )
    for d_layer in discriminator:
      self._discriminator . add ( d_layer )
    self._discriminator . add ( Dense (units = critic_dim) )

  def compile ( self , 
                d_optimizer ,
                g_optimizer , 
                d_updt_per_batch = 1 ,
                g_updt_per_batch = 1 ,
                grad_penalty = 10 ) -> None:
    """Configure the models for CramerGAN training.
    
    Parameters
    ----------
    d_optimizer : `tf.keras.optimizers.Optimizer`
      ...

    g_optimizer : `tf.keras.optimizers.Optimizer`
      ...

    d_updt_per_batch : `int`, optional
      ... (`1`, by default).

    g_updt_per_batch : `int`, optional
      ... (`1`, by default).

    grad_penalty : `float`, optional
      ... (`0.001`, by default).
    """
    super(CramerGAN, self) . compile ( d_optimizer = d_optimizer , 
                                       g_optimizer = g_optimizer ,
                                       d_updt_per_batch = d_updt_per_batch ,
                                       g_updt_per_batch = g_updt_per_batch )
    self._critic = Critic ( lambda x : self._discriminator(x) )

    ## Data-type control
    try:
      grad_penalty = float ( grad_penalty )
    except:
      raise TypeError ("The loss gradient penalty should be a float.")

    self._grad_penalty = grad_penalty
  
  def _compute_d_loss (self, gen_sample, ref_sample, weights = None) -> tf.Tensor:
    """Return the discriminator loss.
    
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
    d_loss : `tf.Tensor`
      ...
    """
    ## Data-batch splitting
    batch_size = tf.shape(gen_sample)[0] / 2
    batch_size = tf.cast (batch_size, tf.int32)

    gen_sample_1 , gen_sample_2 = gen_sample[:batch_size] , gen_sample[batch_size:batch_size*2]
    ref_sample = ref_sample[:batch_size]

    ## Discriminator loss computation
    d_loss = self._critic (gen_sample_1, gen_sample_2) - self._critic (ref_sample, gen_sample_2)
    if weights is not None:
      weights_1 , weights_2 = weights[:batch_size] , weights[batch_size:batch_size*2]
      d_loss = weights_1 * weights_2 * d_loss
    d_loss = tf.reduce_mean (d_loss)

    alpha = tf.random.uniform (
                                shape  = (tf.shape(ref_sample)[0], 1) , 
                                minval = 0. , 
                                maxval = 1. ,
                              )
    differences  = gen_sample_1 - ref_sample
    interpolates = ref_sample + alpha * differences
    critic_int = self._critic ( interpolates , gen_sample_2 )
    grad = tf.gradients ( critic_int , interpolates )
    grad = tf.concat  ( grad , axis = 1 )
    grad = tf.reshape ( grad , shape = (tf.shape(grad)[0], -1) )
    slopes  = tf.norm ( grad , axis = 1 )
    gp_term = tf.square ( tf.maximum ( tf.abs (slopes) - 1., 0. ) )
    gp_term = self._grad_penalty * tf.reduce_mean (gp_term)   # gradient penalty
    d_loss += gp_term
    return d_loss

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
    ## Data-batch splitting
    batch_size = tf.shape(gen_sample)[0] / 2
    batch_size = tf.cast (batch_size, tf.int32)

    gen_sample_1 , gen_sample_2 = gen_sample[:batch_size] , gen_sample[batch_size:batch_size*2]
    ref_sample = ref_sample[:batch_size]

    ## Generator loss computation
    g_loss = self._critic (ref_sample, gen_sample_2) - self._critic (gen_sample_1, gen_sample_2)
    if weights is not None:
      weights_1 , weights_2 = weights[:batch_size] , weights[batch_size:batch_size*2]
      g_loss = weights_1 * weights_2 * g_loss
    return tf.reduce_mean (g_loss)

  @property
  def discriminator (self) -> tf.keras.Sequential:
    """The discriminator of the CramerGAN system."""
    return self._discriminator

  @property
  def generator (self) -> tf.keras.Sequential:
    """The generator of the CramerGAN system."""
    return self._generator

  @property
  def critic_dim (self) -> int:
    """The dimension of the critic space."""
    return self._critic_dim
