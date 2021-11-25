#from __future__ import annotations

from lb_pidsim_train.callbacks.schedulers import GanBaseScheduler

class GanExpScheduler (GanBaseScheduler):   # TODO add docstring
  """class description"""
  def __init__(self, factor = 0.1, step = 1):   # TODO add data-type check
    super(GanExpScheduler, self) . __init__()
    self._factor = factor
    self._step = step

  def _compute_scale_factor (self, lr0, epoch):
    return lr0 * self._factor ** (epoch / self._step)
