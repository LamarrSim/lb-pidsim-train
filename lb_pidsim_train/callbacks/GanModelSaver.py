import os
from tensorflow.keras.callbacks import Callback


class GanModelSaver (Callback):
  def __init__ ( self , 
                 name , 
                 dirname = "./models"  , 
                 model_to_save = "all" , 
                 step = None ,
                 verbose = 1 ) -> None:
    super().__init__()
    self._filename = f"{dirname}/{name}"

    if not os.path.exists (f"{self._filename}"):
      os.makedirs (f"{self._filename}")

    if model_to_save not in ["gen", "disc", "all"]:
      raise ValueError ("`model_to_save` should be chosen in ['gen', 'disc', 'all'].")

    self._model_to_save = model_to_save
    self._step = step
    self._verbose = verbose

  def on_epoch_end (self, epoch, logs = None) -> None:
    if self._step is not None:
      if (epoch + 1) % self._step == 0:
        if self._model_to_save == "gen":
          self.model.generator . save ( f"{self._filename}/saved_generator_ep{epoch+1:04d}", save_format = "tf" )
        elif self._model_to_save == "disc":
          self.model.discriminator . save ( f"{self._filename}/saved_discriminator_ep{epoch+1:04d}", save_format = "tf" )
        else:
          self.model.generator . save ( f"{self._filename}/saved_generator_ep{epoch+1:04d}", save_format = "tf" )
          self.model.discriminator . save ( f"{self._filename}/saved_discriminator_ep{epoch+1:04d}", save_format = "tf" )
    else: pass

  def on_train_end (self, logs = None ) -> None:
    if self._model_to_save == "gen":
      self.model.generator . save ( f"{self._filename}/saved_generator", save_format = "tf" )
      self._verbose_message ( model = "generator", verbose = self._verbose )
    elif self._model_to_save == "disc":
      self.model.discriminator . save ( f"{self._filename}/saved_discriminator", save_format = "tf" )
      self._verbose_message ( model = "discriminator", verbose = self._verbose )
    else:
      self.model.generator . save ( f"{self._filename}/saved_generator", save_format = "tf" )
      self._verbose_message ( model = "generator", verbose = self._verbose )
      self.model.discriminator . save ( f"{self._filename}/saved_discriminator", save_format = "tf" )
      self._verbose_message ( model = "discriminator", verbose = self._verbose )

  def _verbose_message (self, model, verbose = 1) -> None:
    print (f"[INFO] Trained {model} correctly exported to {self._filename}/saved_{model}")
