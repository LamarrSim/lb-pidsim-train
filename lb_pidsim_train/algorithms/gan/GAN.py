#from __future__ import annotations

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


d_loss_tracker = tf.keras.metrics.Mean ( name = "d_loss" )
"""Metric instance to track the discriminator loss score."""

g_loss_tracker = tf.keras.metrics.Mean ( name = "g_loss" )
"""Metric instance to track the generator loss score."""

mse_tracker = tf.keras.metrics.MeanSquaredError ( name = "mse" )
"""Metric instance to track the mean square error."""


class GAN (tf.keras.Model):
  """Keras model class to build and train GAN system.
  
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
  def __init__ ( self , 
                 X_shape ,
                 Y_shape ,
                 discriminator ,
                 generator     , 
                 latent_dim = 64 ) -> None:
    super(GAN, self) . __init__()
    self._loss_name = "Loss function"

    ## Feature space dimension
    if isinstance ( X_shape, (tuple, list, np.ndarray, tf.Tensor) ):
      X_shape = int ( X_shape[1] )
    if isinstance ( Y_shape, (tuple, list, np.ndarray, tf.Tensor) ):
      Y_shape = int ( Y_shape[1] )

    self._X_shape = X_shape
    self._Y_shape = Y_shape

    ## Data-type control
    try:
      latent_dim = int ( latent_dim )
    except:
      raise TypeError ("The latent space dimension should be an integer.")

    self._latent_dim = latent_dim

    ## Discriminator sequential model
    self._discriminator = Sequential ( name = "discriminator" )
    for d_layer in discriminator:
      self._discriminator . add ( d_layer )
    self._discriminator . add ( Dense ( units = 1 , 
                                        activation = "sigmoid" , 
                                        kernel_initializer = "he_normal" ) )

    ## Generator sequential model
    self._generator = Sequential ( name = "generator" )
    for g_layer in generator:
      self._generator . add ( g_layer )
    self._generator . add ( Dense (units = Y_shape) )

  def compile ( self , 
                d_optimizer ,
                g_optimizer , 
                d_updt_per_batch = 1 ,
                g_updt_per_batch = 1 ) -> None:
    """Configure the models for GAN training.
    
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
    """
    super(GAN, self) . compile()

    ## Build discriminator and generator models
    self._discriminator . build ( input_shape = (None, self._X_shape + self._Y_shape) )
    self._generator . build ( input_shape = (None, self._X_shape + self._latent_dim) )

    self._d_optimizer = d_optimizer
    self._g_optimizer = g_optimizer

    self._d_lr0 = d_optimizer.learning_rate
    self._g_lr0 = g_optimizer.learning_rate

    ## Data-type control
    try:
      d_updt_per_batch = int ( d_updt_per_batch )
    except:
      raise TypeError ("The number of discriminator updates per batch should be an integer.")
    try:
      g_updt_per_batch = int ( g_updt_per_batch )
    except:
      raise TypeError ("The number of generator updates per batch should be an integer.")

    ## Data-value control
    if d_updt_per_batch == 0:
      raise ValueError ("The number of discriminator updates per batch should be greater than 0.")
    if g_updt_per_batch == 0:
      raise ValueError ("The number of generator updates per batch should be greater than 0.")

    self._d_updt_per_batch = d_updt_per_batch
    self._g_updt_per_batch = g_updt_per_batch

  def summary (self) -> None:
    """Print a string summary of the discriminator and generator networks."""
    self._discriminator . summary()
    self._generator . summary()

  @staticmethod
  def _unpack_data (data):
    """Unpack data-batch into input, output and weights (`None`, if not available)."""
    if len(data) == 3:
      X , Y , w = data
    else:
      X , Y = data
      w = None
    return X, Y, w

  def train_step (self, data) -> dict:
    """Train step for Keras APIs."""
    X, Y, w = self._unpack_data (data)

    ## Discriminator updates per batch
    for i in range(self._d_updt_per_batch):
      self._train_d_step (X, Y, w)

    ## Generator updates per batch
    for j in range(self._g_updt_per_batch):
      self._train_g_step (X, Y, w)

    ## Loss computation
    ref_sample, gen_sample = self._arrange_samples (X, Y)
    d_loss = self._compute_d_loss (gen_sample, ref_sample, weights = w)
    g_loss = self._compute_g_loss (gen_sample, ref_sample, weights = w)

    ## Update metrics state
    d_loss_tracker . update_state (d_loss)
    g_loss_tracker . update_state (g_loss)

    Y_gen = self.generate (X)
    mse_tracker . update_state (Y, Y_gen, sample_weight = w)

    return { "mse"    : mse_tracker.result()    ,
             "d_loss" : d_loss_tracker.result() , 
             "g_loss" : g_loss_tracker.result() ,
             "d_lr"   : self._d_optimizer.lr    ,
             "g_lr"   : self._g_optimizer.lr    }

  def test_step (self, data) -> dict:
    """Test step for Keras APIs."""
    X, Y, w = self._unpack_data (data)

    ## Loss computation
    ref_sample, gen_sample = self._arrange_samples (X, Y)
    d_loss = self._compute_d_loss (gen_sample, ref_sample, weights = w)
    g_loss = self._compute_g_loss (gen_sample, ref_sample, weights = w)

    ## Update metrics state
    d_loss_tracker . update_state (d_loss)
    g_loss_tracker . update_state (g_loss)

    Y_gen = self.generate (X)
    mse_tracker . update_state (Y, Y_gen, sample_weight = w)

    return { "mse"    : mse_tracker.result()    ,
             "d_loss" : d_loss_tracker.result() , 
             "g_loss" : g_loss_tracker.result() ,
             "d_lr"   : self._d_optimizer.lr    ,
             "g_lr"   : self._g_optimizer.lr    }

  def _arrange_samples (self, X, Y) -> tuple:
    """Arrange the reference and generated samples.
    
    Parameters
    ----------
    X : `tf.Tensor`
      ...

    Y : `tf.Tensor`
      ...
    
    Returns
    -------
    ref_sample : `tf.Tensor`
      ...

    gen_sample : `tf.Tensor`
      ...
    """
    ## Sample random points in the latent space
    batch_size = tf.shape(X)[0]
    latent_vectors = tf.random.normal ( shape = (batch_size, self._latent_dim) )

    ## Map the latent space into the generated space
    input_vectors = tf.concat ( [X, latent_vectors], axis = 1 )
    generated = self._generator (input_vectors)

    ## Reference and generated sample
    ref_sample = tf.concat ( [X, Y], axis = 1 )
    gen_sample = tf.concat ( [X, generated], axis = 1 )
    return ref_sample, gen_sample

  def _train_d_step (self, X, Y, w = None) -> None:
    """Training step for the discriminator.
    
    Parameters
    ----------
    X : `tf.Tensor`
      ...

    Y : `tf.Tensor`
      ...

    w : `tf.Tensor`, optional
      ... (`None`, by default).
    """
    with tf.GradientTape() as tape:
      ref_sample, gen_sample = self._arrange_samples (X, Y)
      d_loss = self._compute_d_loss ( gen_sample, ref_sample, weights = w )
    grads = tape.gradient ( d_loss, self._discriminator.trainable_weights )
    self._d_optimizer.apply_gradients ( zip (grads, self._discriminator.trainable_weights) )

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
    return - self._compute_g_loss (gen_sample, ref_sample, weights)

  def _train_g_step (self, X, Y, w = None) -> None:
    """Training step for the generator.
    
    Parameters
    ----------
    X : `tf.Tensor`
      ...

    Y : `tf.Tensor`
      ...

    w : `tf.Tensor`, optional
      ... (`None`, by default).
    """
    with tf.GradientTape() as tape:
      ref_sample, gen_sample = self._arrange_samples (X, Y)
      g_loss = self._compute_g_loss ( gen_sample, ref_sample, weights = w )
    grads = tape.gradient ( g_loss, self._generator.trainable_weights )
    self._g_optimizer.apply_gradients ( zip (grads, self._generator.trainable_weights) )

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
    ## Noise injection to stabilize GAN training
    rnd_gen = tf.random.normal ( tf.shape(gen_sample), mean = 0., stddev = 0.1 )
    rnd_ref = tf.random.normal ( tf.shape(ref_sample), mean = 0., stddev = 0.1 )
    D_gen = self._discriminator ( gen_sample + rnd_gen )
    D_ref = self._discriminator ( ref_sample + rnd_ref )

    ## Loss computation
    g_loss = tf.math.log ( tf.clip_by_value (D_ref, 1e-12, 1.) * tf.clip_by_value (1 - D_gen, 1e-12, 1.) )
    if weights is not None:
      g_loss = weights * g_loss
    return tf.reduce_mean (g_loss)

  def generate (self, X) -> tf.Tensor:
    """Method to generate the target variables `Y` given the input features `X`.
    
    Parameters
    ----------
    X : `tf.Tensor`
      ...

    Returns
    -------
    Y_gen : `tf.Tensor`
      ...
    """
    ## Sample random points in the latent space
    batch_size = tf.shape(X)[0]
    latent_dim = self.latent_dim
    latent_tensor = tf.random.normal ( shape = (batch_size, latent_dim) )

    ## Map the latent space into the generated space
    input_tensor = tf.concat ( [X, latent_tensor], axis = 1 )
    Y_gen = self.generator (input_tensor)
    return Y_gen

  @property
  def loss_name (self) -> str:
    """Name of the loss function used for training."""
    return self._loss_name

  @property
  def discriminator (self) -> tf.keras.Sequential:
    """The discriminator of the GAN system."""
    return self._discriminator

  @property
  def generator (self) -> tf.keras.Sequential:
    """The generator of the GAN system."""
    return self._generator

  @property
  def latent_dim (self) -> int:
    """The dimension of the latent space."""
    return self._latent_dim

  @property
  def d_optimizer (self) -> tf.keras.optimizers.Optimizer:
    """The discriminator optimizer."""
    return self._d_optimizer

  @property
  def g_optimizer (self) -> tf.keras.optimizers.Optimizer:
    """The generator optimizer.."""
    return self._g_optimizer

  @property
  def d_lr0 (self) -> float:
    """Initial value for discriminator learning rate."""
    return self._d_lr0

  @property
  def g_lr0 (self) -> float:
    """Initial value for generator learning rate."""
    return self._g_lr0

  @property
  def metrics (self) -> list:
    return [d_loss_tracker, g_loss_tracker, mse_tracker]
