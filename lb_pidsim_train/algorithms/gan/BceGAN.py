#from __future__ import annotations

import tensorflow as tf
from lb_pidsim_train.algorithms.gan import GAN


class BceGAN (GAN):   # TODO add class description
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
  def __init__ ( self ,
                 X_shape ,
                 Y_shape ,
                 discriminator ,
                 generator     ,
                 referee = None ,
                 latent_dim = 64 ) -> None:
    super().__init__ ( X_shape = X_shape ,
                       Y_shape = Y_shape ,
                       discriminator = discriminator , 
                       generator     = generator     ,
                       referee       = referee       ,
                       latent_dim    = latent_dim    )
    self._loss_name = "Binary cross entropy"

  def compile ( self , 
                d_optimizer , 
                g_optimizer ,
                r_optimizer = None ,
                d_updt_per_batch = 1 , 
                g_updt_per_batch = 1 ) -> None:   # TODO complete docstring
    """Configure the models for BceGAN training.
    
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
    super().compile ( d_optimizer = d_optimizer , 
                      g_optimizer = g_optimizer , 
                      r_optimizer = r_optimizer ,
                      d_updt_per_batch = d_updt_per_batch , 
                      g_updt_per_batch = g_updt_per_batch )

    self._k_gen = 0.1
    self._k_ref = 0.9

  def _compute_g_loss (self, gen_sample, ref_sample) -> tf.Tensor:   # TODO complete docstring
    """Return the generator loss.
    
    Parameters
    ----------
    gen_sample : `tuple` of `tf.Tensor`
      ...

    ref_sample : `tuple` of `tf.Tensor`
      ...

    Returns
    -------
    g_loss : `tf.Tensor`
      ...
    """
    ## Extract input tensors and weights
    XY_gen, w_gen = gen_sample
    XY_ref, w_ref = ref_sample

    ## Noise injection to stabilize BceGAN training
    rnd_gen = tf.random.normal ( tf.shape(XY_gen), mean = 0., stddev = 0.05 )
    rnd_ref = tf.random.normal ( tf.shape(XY_ref), mean = 0., stddev = 0.05 )
    D_gen = self._discriminator ( XY_gen + rnd_gen )
    D_ref = self._discriminator ( XY_ref + rnd_ref )

    ## Loss computation
    g_loss = w_gen * self._k_gen       * tf.math.log ( tf.clip_by_value ( D_gen     , 1e-12 , 1.0 ) ) + \
             w_gen * (1 - self._k_gen) * tf.math.log ( tf.clip_by_value ( 1 - D_gen , 1e-12 , 1.0 ) ) + \
             w_ref * self._k_ref       * tf.math.log ( tf.clip_by_value ( D_ref     , 1e-12 , 1.0 ) ) + \
             w_ref * (1 - self._k_ref) * tf.math.log ( tf.clip_by_value ( 1 - D_ref , 1e-12 , 1.0 ) ) 
    return tf.reduce_mean (g_loss)

  def _compute_threshold (self, ref_sample) -> tf.Tensor:   # TODO complete docstring
    """Return the threshold for loss values.
    
    Parameters
    ----------
    ref_sample : `tuple` of `tf.Tensor`
      ...

    Returns
    -------
    th_loss : `tf.Tensor`
      ...
    """
    ## Extract input tensors and weights
    XY_ref, w_ref = ref_sample

    ## Noise injection to stabilize GAN training
    rnd_ref = tf.random.normal ( tf.shape(XY_ref), mean = 0., stddev = 0.05 )
    D_ref = self._discriminator ( XY_ref + rnd_ref )

    ## Split tensors and weights
    batch_size = tf.cast ( tf.shape(XY_ref)[0] / 2, tf.int32 )
    D_ref_1, D_ref_2 = D_ref[:batch_size], D_ref[batch_size:batch_size*2]
    w_ref_1, w_ref_2 = w_ref[:batch_size], w_ref[batch_size:batch_size*2]

    ## Threshold loss computation
    th_loss = w_ref_1 * self._k_gen       * tf.math.log ( tf.clip_by_value ( D_ref_1     , 1e-12 , 1.0 ) ) + \
              w_ref_1 * (1 - self._k_gen) * tf.math.log ( tf.clip_by_value ( 1 - D_ref_1 , 1e-12 , 1.0 ) ) + \
              w_ref_2 * self._k_ref       * tf.math.log ( tf.clip_by_value ( D_ref_2     , 1e-12 , 1.0 ) ) + \
              w_ref_2 * (1 - self._k_ref) * tf.math.log ( tf.clip_by_value ( 1 - D_ref_2 , 1e-12 , 1.0 ) ) 
    return tf.reduce_mean (th_loss)

  @property
  def discriminator (self) -> tf.keras.Sequential:
    """The discriminator of the BceGAN system."""
    return self._discriminator

  @property
  def generator (self) -> tf.keras.Sequential:
    """The generator of the BceGAN system."""
    return self._generator

  @property
  def k_gen (self) -> float:
    """Smoothness weight of the BCE for the reference dataset."""
    return self._k_gen

  @k_gen.setter
  def k_gen (self, k) -> None:
    ## Data-type control
    try:
      k_gen = float (k_gen)
    except: 
      raise TypeError ("The smoothness weight should be a float.")

    ## Data-value control
    if (k_gen < 0.0) or (k_gen > 1.0):
      raise ValueError ("The smoothness weight should be between 0 and 1.")

    self._k_gen = k

  @property
  def k_ref (self) -> float:
    """Smoothness weight of the BCE for the generated dataset."""
    return self._k_ref

  @k_ref.setter
  def k_ref (self, k) -> None:
    ## Data-type control
    try:
      k_ref = float (k_ref)
    except: 
      raise TypeError ("The smoothness weight should be a float.")

    ## Data-value control
    if (k_ref < 0.0) or (k_ref > 1.0):
      raise ValueError ("The smoothness weight should be between 0 and 1.")

    self._k_ref = k
    