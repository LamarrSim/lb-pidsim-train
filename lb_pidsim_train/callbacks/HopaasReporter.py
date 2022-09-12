from time import time
from tensorflow.keras.callbacks import Callback


class HopaasReporter (Callback):
  def __init__ (self, trial, loss, pruning = False, step = None, timeout = None) -> None:
    # TODO: add docstring
    # NOTE: the step value is strongly related to interval_steps in the Optuna pruner function
    # NOTE: too small step value can produce overhead
    super().__init__()
    self._trial = trial
    self._loss = loss
    self._pruning = pruning
    self._step = step
    self._timeout = timeout

  def on_train_begin (self, logs = None) -> None:
    if self._timeout is not None:
      self._timer = time()

  def on_epoch_end (self, epoch, logs = None) -> None:
    if self._timeout is not None:
      train_time = int ( time() - self._timer )
      if train_time >= self._timeout:
        self.model.stop_training = True

    if self._step is not None:
      if (epoch + 1) % self._step == 0:
        if self._pruning:
          self._trial.loss = self._get_monitor_value (logs = logs)
          if self._trial.should_prune:
            self.model.stop_training = True

  def on_train_end (self, logs = None) -> None:
    if not self._pruning:
      self._trial.loss = self._get_monitor_value (logs = logs)
    else:
      if self._trial.should_prune:
        print ( f"[INFO] Training trial n. {self._trial.id} "
                 "remotely pruned by the Hopaas server" )

  def _get_monitor_value (self, logs) -> float:
    logs = logs or {}
    monitor_value = float ( logs.get(self._loss) )
    if monitor_value is None:
      monitor_keys = list ( logs.keys() )
      print ( f"[WARNING] Pruning strategy conditioned on metric `{self._loss}` which "
              f"is not available. Available metrics are: {','.join(monitor_keys)}." )
    return monitor_value
