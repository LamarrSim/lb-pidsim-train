from time import time
from tensorflow.keras.callbacks import Callback


class StopWatch (Callback):
  def __init__ (self, timeout = None) -> None:
    super().__init__()
    self._timeout = timeout

  def on_train_begin (self, logs = None) -> None:
    if self._timeout is not None:
      self._timer = time()

  def on_epoch_end(self, epoch, logs = None) -> None:
    if self._timeout is not None:
      train_time = int ( time() - self._timer )
      if train_time >= self._timeout:
        self.model.stop_training = True