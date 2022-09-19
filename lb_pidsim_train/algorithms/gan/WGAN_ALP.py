#from __future__ import annotations

import inspect
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from lb_pidsim_train.algorithms.gan import GAN


class WGAN_ALP (GAN):   # TODO add class description
  """Keras model class to build and train WGAN-ALP system.
  
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
    self._loss_name = "Wasserstein distance"

    ## Discriminator sequential model
    self._discriminator = Sequential ( name = "discriminator" )
    for d_layer in discriminator:
      self._discriminator . add ( d_layer )
    self._discriminator . add ( Dense ( units = 1, activation = "linear" , 
                                        kernel_initializer = "glorot_normal" ) )

  def compile ( self , 
                d_optimizer , 
                g_optimizer ,
                r_optimizer = None ,
                d_updt_per_batch = 1 , 
                g_updt_per_batch = 1 ,
                v_adv_dir_updt   = 1 , 
                adv_lp_penalty = 100 ) -> None:
    """Configure the models for WGAN-ALP training.
    
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

    v_adv_dir_updt : `int`, optional
      ... (`1`, by default).

    adv_lp_penalty : `int`, optional
      ... (`100`, by default).
    """
    super().compile ( d_optimizer = d_optimizer , 
                      g_optimizer = g_optimizer , 
                      r_optimizer = r_optimizer ,
                      d_updt_per_batch = d_updt_per_batch , 
                      g_updt_per_batch = g_updt_per_batch )

    ## Data-type control
    if not isinstance (v_adv_dir_updt, int):
      if isinstance (v_adv_dir_updt, float): v_adv_dir_updt = int (v_adv_dir_updt)
      else: raise TypeError ("The number of virtual adversarial direction updates should be an integer.")

    if not isinstance (adv_lp_penalty, float):
      if isinstance (adv_lp_penalty, int): adv_lp_penalty = float (adv_lp_penalty)
      else: raise TypeError ("The adversarial Lipschitz penalty should be a float.")

    ## Data-value control
    if v_adv_dir_updt <= 0:
      raise ValueError ("The number of virtual adversarial direction updates should be greater than 0.")

    if adv_lp_penalty <= 0:
      raise ValueError ("The adversarial Lipschitz penalty should be greater than 0.")

    self._v_adv_dir_updt = v_adv_dir_updt
    self._adv_lp_penalty = adv_lp_penalty
    self._lp_const = 1.0
    self._epsilon_sampler = lambda shape, dtype: tf.math.exp ( tf.random.uniform ( shape  = shape ,
                                                                                   minval = tf.math.log (1e-2) ,
                                                                                   maxval = tf.math.log (1e+0) ,
                                                                                   dtype  = dtype ) )
    self._xi = 10.0

  def _compute_d_loss (self, gen_sample, ref_sample) -> tf.Tensor:   # TODO complete docstring
    """Return the discriminator loss.
    
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

    ## Standard WGAN loss
    D_gen = self._discriminator ( XY_gen )
    D_ref = self._discriminator ( XY_ref )
    d_loss = tf.reduce_mean ( w_gen * D_gen - w_ref * D_ref , axis = None )

    ## Initial virtual adversarial direction
    r_k = tf.random.uniform ( shape  = (tf.shape(XY_ref)[1],) , 
                              minval = 0.0 , 
                              maxval = 1.0 , 
                              dtype  = XY_ref.dtype )
    r_k /= tf.norm ( r_k , axis = None )

    for _ in range(self._v_adv_dir_updt):
      ## adversarial perturbation of input tensors
      XY_gen_pert = tf.clip_by_value ( XY_gen + self._xi * r_k , 
                                       clip_value_min = tf.reduce_min (XY_gen, axis = 0) ,
                                       clip_value_max = tf.reduce_max (XY_gen, axis = 0) )
      XY_ref_pert = tf.clip_by_value ( XY_ref + self._xi * r_k , 
                                       clip_value_min = tf.reduce_min (XY_ref, axis = 0) ,
                                       clip_value_max = tf.reduce_max (XY_ref, axis = 0) )

      ## approximation of virtual adversarial direction
      D_gen_pert = tf.cast ( self._discriminator (XY_gen_pert), dtype = XY_gen.dtype )
      D_ref_pert = tf.cast ( self._discriminator (XY_ref_pert), dtype = XY_ref.dtype )
      diff = tf.abs ( tf.concat ( [ D_gen      , D_ref      ] , axis = 0 ) - \
                      tf.concat ( [ D_gen_pert , D_ref_pert ] , axis = 0 ) )
      diff = tf.reduce_mean ( diff, axis = None )
      r_k = tf.gradients ( diff, [r_k] ) [0]
      r_k /= tf.norm ( r_k , axis = None )

    ## virtual adversarial direction
    epsilon = self._epsilon_sampler ( shape = (tf.shape(XY_ref)[1],), dtype = XY_ref.dtype )
    r_adv = epsilon * r_k

    ## adversarial perturbation of input tensors
    XY_gen_pert = tf.clip_by_value ( XY_gen + r_adv , 
                                     clip_value_min = tf.reduce_min (XY_gen, axis = 0) ,
                                     clip_value_max = tf.reduce_max (XY_gen, axis = 0) )
    XY_ref_pert = tf.clip_by_value ( XY_ref + r_adv , 
                                     clip_value_min = tf.reduce_min (XY_ref, axis = 0) ,
                                     clip_value_max = tf.reduce_max (XY_ref, axis = 0) )

    ## adversarial Lipschitz penalty correction
    D_gen_pert = tf.cast ( self._discriminator (XY_gen_pert), dtype = XY_gen.dtype )
    D_ref_pert = tf.cast ( self._discriminator (XY_ref_pert), dtype = XY_ref.dtype )
    diff = tf.abs ( tf.concat ( [ D_gen      , D_ref      ] , axis = 0 ) - \
                    tf.concat ( [ D_gen_pert , D_ref_pert ] , axis = 0 ) )
    alp_term = tf.math.maximum ( diff / tf.norm ( r_adv, axis = None ) - self._lp_const, 0.0 )   # one-side penalty
    alp_term = self._adv_lp_penalty * tf.reduce_mean (alp_term, axis = None)   # adversarial Lipschitz penalty
    d_loss += alp_term ** 2
    return d_loss

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

    ## Standard WGAN loss
    D_gen = self._discriminator ( XY_gen )
    D_ref = self._discriminator ( XY_ref )
    g_loss = w_ref * D_ref - w_gen * D_gen
    return tf.reduce_mean (g_loss, axis = None)

  def _compute_threshold (self, ref_sample) -> tf.Tensor:
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
    _, w_ref = ref_sample
    th_loss = tf.zeros_like (w_ref)
    return tf.reduce_sum (th_loss, axis = None)

  @property
  def discriminator (self) -> tf.keras.Sequential:
    """The discriminator of the WGAN-ALP system."""
    return self._discriminator

  @property
  def generator (self) -> tf.keras.Sequential:
    """The generator of the WGAN-ALP system."""
    return self._generator

  @property
  def v_adv_dir_updt (self) -> int:
    """Number of recursive updates of the virtual adversarial direction."""
    return self._v_adv_dir_updt

  @property
  def adv_lp_penalty (self) -> float:
    """Adversarial Lipschitz penalty coefficient."""
    return self._adv_lp_penalty

  @property
  def lp_const (self) -> float:
    """Lipschitz constant of the discriminator."""
    return self._lp_const

  @lp_const.setter
  def lp_const (self, K) -> None:
    ## data-type control
    if not isinstance (K, float):
      if isinstance (K, int): K = float (K)
      else: raise TypeError ("The Lipschitz constant should be a float.")

    ## data-value control
    if K <= 0: 
      raise ValueError ("The Lipschitz constant should be greater than 0.")

    self._lp_const = K

  @property
  def epsilon_sampler (self) -> str:
    """The definition of the sampler function for the epsilon hyperparameter."""
    return inspect.getsource (self._epsilon_sampler)

  @epsilon_sampler.setter
  def epsilon_sampler (self, func) -> None:
    ## data-type control
    if not ( callable(func) and func.__name__ == "<lambda>" ):
      raise TypeError ("The epsilon sampler should be passed as a lambda function.")

    ## data-value control
    func_args = func.__code__.co_varnames
    if (len(func_args) != 2) or (func_args[0] != "shape") or (func_args[1] != "dtype"): 
      raise ValueError ( f"The lambda function for the epsilon sampler "
                         f"should have only ('shape', 'dtype') as arguments." )

    self._epsilon_sampler = func

  @property
  def xi (self) -> float:
    """The xi hyperparameter used to approximate the virtual adversarial direction."""
    return self._xi

  @xi.setter
  def xi (self, xi) -> None:
    ## data-type control
    if not isinstance (xi, float):
      if isinstance (xi, int): xi = float (xi)
      else: raise TypeError ("The xi hyperparameter should be a float.")

    ## data-value control
    if xi <= 0: 
      raise ValueError ("The xi hyperparameter should be greater than 0.")

    self._xi = xi
