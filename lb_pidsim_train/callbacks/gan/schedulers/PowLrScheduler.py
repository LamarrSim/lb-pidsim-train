#from __future__ import annotations

from lb_pidsim_train.callbacks.gan.schedulers import BaseLrScheduler

class PowLrScheduler (BaseLrScheduler):   # TODO add docstring
  """class description"""
  def __init__ (self, decay = 1, step = 1):   # TODO add data-type check
    super().__init__()
    self._decay = decay
    self._step = step

  def _scheduled_lr (self, lr0, epoch):
    return lr0 / (1.0 + self._decay * epoch / self._step)
