import os
import shutil
from tensorflow.keras.callbacks import Callback


class HopaasModelSaver (Callback):
  def __init__ ( self  , 
                 trial ,
                 name  , 
                 dirname = "./models"   , 
                 model_to_save = "gen"  ,
                 min_trials_to_save = 0 ,
                 verbose = 1 ) -> None:
    super().__init__()
    self._trial = trial
    self._filename = f"{dirname}/{name}"

    if not os.path.exists (f"{self._filename}"):
      os.makedirs (f"{self._filename}")

    if model_to_save not in ["gen", "disc", "all"]:
      raise ValueError ("`model_to_save` should be chosen in ['gen', 'disc', 'all'].")

    self._model_to_save = model_to_save
    self._min_trials_to_save = min_trials_to_save
    self._verbose = verbose

  def on_train_end (self, logs = None ) -> None:
    if (self._trial.id >= self._min_trials_to_save) and (not self._trial.should_prune):
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
    else:
      shutil.rmtree ( self._filename, ignore_errors = True )
      if self._verbose > 0: print (f"[INFO] Export directory for trial n. {self._trial.id} correctly deleted")

  def _verbose_message (self, model, verbose = 1) -> None:
    if verbose > 0: print (f"[INFO] Trained {model} correctly exported to {self._filename}/saved_{model}")
