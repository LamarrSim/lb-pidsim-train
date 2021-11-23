#from __future__ import annotations

from lb_pidsim_train.callbacks.schedulers import GanBaseScheduler

class GanPowScheduler (GanBaseScheduler):   # TODO add docstring
  """class description"""
  def __init__(self, decay = 1, step = 1):   # TODO add data-type check
    super(GanPowScheduler, self) . __init__()
    self._decay = decay
    self._step = step

  def _compute_scale_factor (self, epoch):
    return 1.0 / (1.0 + self._decay / self._step)
