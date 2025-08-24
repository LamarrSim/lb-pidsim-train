import yaml
import time
from tensorflow.keras.callbacks import Callback
from pidgan.players import generators, discriminators, classifiers
import os.path

class TimeLimitCallback(Callback):
    """Stop training if it exceeds a specified time limit (in seconds)."""
    def __init__(self, max_duration_seconds):
        super().__init__()
        self.max_duration = max_duration_seconds
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.start_time
        if elapsed > self.max_duration:
            print(f"\n\nTraining stopped after {elapsed:.2f}s (limit: {self.max_duration}s)")
            self.model.stop_training = True


def create_models(
    model: str,
    particle: str,
    output_dim: int,
    model_filename: str = None,
    kwargs_generator=None,
    kwargs_discriminator=None,
    kwargs_referee=None,
):
    if model_filename is None:
        model_filename = os.path.join(os.path.dirname(__file__), "models.yaml")

    with open(model_filename) as model_file:
        full_config=yaml.safe_load(model_file)

    cfg = full_config.get(model, {}).get(particle)
    if cfg is None:
        print(full_config)
        raise KeyError(
            f"Failed loading model for {model} {particle} from {model_filename}"
        )

    generator = None
    if 'generator' in cfg.keys():
        Generator = getattr(generators, cfg['generator']['class'])
        generator = Generator(
            output_dim=output_dim,
            name='generator',
            **(cfg['generator']['params']),
            **(kwargs_generator if kwargs_generator is not None else {}),
        )

    discriminator = None
    if 'discriminator' in cfg.keys():
        Discriminator = getattr(discriminators, cfg['discriminator']['class'])
        discriminator = Discriminator(
            output_dim=1,
            name="discriminator",
            **(cfg['discriminator']['params']),
            **(kwargs_discriminator if kwargs_discriminator is not None else {}),
       )

    referee = None
    if 'referee' in cfg.keys():
        Referee = getattr(classifiers, cfg['referee']['class'])
        referee = Referee(
            name="referee",
            **(cfg['referee']['params']),
            **(kwargs_referee if kwargs_referee is not None else {}),
        )

    return generator, discriminator, referee












    