import tensorflow as tf
from tensorflow.keras.layers import Layer


class Sampling(Layer):
    """Reparameterization trick: samples z from N(z_mean, exp(z_log_var/2)).

    Works with any tensor shape (2D for global latent, 3D for per-timestep latent).
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
