from tensorflow.keras.callbacks import Callback


class CyclicInitializer (Callback):
  def __init__ (self, cycle = 1) -> None:
    super().__init__()
    self._cycle = cycle

  def on_epoch_end (self, epoch, logs = None) -> None:
    if (epoch + 1) % self._cycle == 0:
      for layer in self.model.classifier.layers:    
        if hasattr (layer, "kernel_initializer") and hasattr (layer, "bias_initializer"):
          layer . set_weights ( [ 
                                  layer.kernel_initializer ( shape = layer.kernel.shape ) , 
                                  layer.bias_initializer   ( shape = layer.bias.shape   )
                              ] )
