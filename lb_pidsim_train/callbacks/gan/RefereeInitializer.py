from tensorflow.keras.callbacks import Callback


class RefereeInitializer (Callback):
  def __init__ (self, step = None) -> None:
    super().__init__()
    self._step = step

  def on_epoch_end (self, epoch, logs = None) -> None:
    if self._step is not None:
      if (epoch + 1) % self._step == 0:
        for layer in self.model.referee.layers:    
          if hasattr (layer, "kernel_initializer") and hasattr (layer, "bias_initializer"):
            layer . set_weights ( [ 
                                    layer.kernel_initializer ( shape = layer.kernel.shape ) , 
                                    layer.bias_initializer   ( shape = layer.bias.shape   )
                                ] )
