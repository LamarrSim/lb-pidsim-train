#from __future__ import annotations

import numpy as np
from lb_pidsim_train.utils import getBinPDFs


def KS_test ( x_gen , 
              x_ref , 
              bins  = 100  , 
              w_gen = None , 
              w_ref = None ) -> np.ndarray:
  """Return the Kolmogorov–Smirnov test of the two input datasets.

  Parameters
  ----------
  x_gen : array_like
    Array containing the generated dataset.

  x_ref : array_like
    Array containing the reference dataset.

  bins : `int`, optional
    Number of equal-width bins in the computed range used to approximate 
    the two PDFs with binned data (`100`, by default).

  w_gen : `int` or `float` or array_like, optional
    An array of weights, of the same length as `x_gen`. Each value in `x_gen` 
    only contributes its associated weight towards the bin count (instead of 1).

  w_ref : `int` or `float` or array_like, optional
    An array of weights, of the same length as `x_ref`. Each value in `x_ref` 
    only contributes its associated weight towards the bin count (instead of 1).

  Returns
  -------
  ks_test : `np.ndarray`
    Array containing the K-S test for each feature of the two input datasets.

  See Also
  --------
  lb_pidsim_train.utils.getBinPDFs : 
    Internal function used to compute the binned PDFs of the input datasets.

  Notes
  -----
  In statistics, the **Kolmogorov–Smirnov test** (K–S test) is a nonparametric 
  test of the equality of one-dimensional probability distributions that can be 
  used to compare two samples.

  Let :math:`G(x)` be the cumulative distribution function of the generated
  dataset and :math:`R(x)` be the one of the reference dataset, then the
  Kolmogorov-Smirnov statistic is 

  .. math::

    D_{KS} = \sup_{x} \left | G(x) - R(x) \right |,

  where :math:`\sup` is the supremum function.

  Examples
  --------
  >>> import numpy as np
  >>> a = np.random.normal ( 0. , 1., 10000 )
  >>> b = np.random.normal ( 0.5, 1., 10000 )
  >>> from lb_pidsim_train.metrics import KS_test
  >>> KS_test ( a, b )
  [0.196027]
  """
  ## Binned PDFs
  p , q = getBinPDFs (x_gen, x_ref, bins, w_gen, w_ref)

  ## Promotion to 2-D arrays
  if len (p.shape) == 1:
    p = p [:, np.newaxis]
  if len (q.shape) == 1:
    q = q [:, np.newaxis]

  ## CDFs computation
  G = np.cumsum (p, axis = 1)
  R = np.cumsum (q, axis = 1)

  ## K-S test computation
  return np.absolute (G - R) . max (axis = 1)



if __name__ == "__main__":
  ## SAMPLE N. 1
  gauss_1 = np.random.normal  ( 0.   , 1.  , size = int(1e6) )
  unif_1  = np.random.uniform ( -0.5 , 0.5 , size = int(1e6) )
  sample_1 = np.c_ [gauss_1, unif_1]

  ## SAMPLE N. 2
  gauss_2 = np.random.normal  ( 0.5  , 1.  , size = int(1e6) )
  unif_2  = np.random.uniform ( -0.4 , 0.6 , size = int(1e6) )
  sample_2 = np.c_ [gauss_2, unif_2]

  binnings = [10, 100, 1000, 10000]
  for bins in binnings:
    ks_test = KS_test (sample_1, sample_2, bins)
    print ( "K-S test (bins - {:.2e}) : {}" . format (bins, ks_test) )
  