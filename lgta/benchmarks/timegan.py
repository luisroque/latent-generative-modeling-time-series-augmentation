from os import path

from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

DEFAULT_TIMEGAN_WEIGHTS_PATH = "assets/model_weights/timegan_tourism.pkl"


def train_timegan_model(
    train_data,
    gan_args=ModelParameters(
        batch_size=128, lr=5e-4, noise_dim=32, layers_dim=128, latent_dim=200, gamma=1
    ),
    train_args=TrainParameters(epochs=500, sequence_length=12, number_sequences=304),
    model_path: str = DEFAULT_TIMEGAN_WEIGHTS_PATH,
):
    """
    Train the TimeGAN model. Model weights are saved under assets/model_weights by default.
    """
    import os

    weights_dir = path.dirname(model_path)
    if weights_dir:
        os.makedirs(weights_dir, exist_ok=True)
    if path.exists(model_path):
        synth = TimeSeriesSynthesizer.load(model_path)
    else:
        synth = TimeSeriesSynthesizer(modelname="timegan", model_parameters=gan_args)
        synth.fit(train_data, train_args, num_cols=train_data.columns.tolist())
        synth.save(model_path)
    return synth


def generate_synthetic_samples(synth, num_samples, detemporalize_func):
    """
    Generate synthetic samples using the trained synthesizer and apply detemporalization.
    """
    synth_data = synth.sample(n_samples=num_samples)
    return detemporalize_func(synth_data)
