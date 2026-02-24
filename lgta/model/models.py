"""
L-GTA model architecture: Conditional Variational Autoencoder (CVAE) with
Bi-LSTM encoder/decoder and Variational Multi-Head Attention (VMHA).

Encoder: Bi-LSTM(z) -> h_t, then VMHA(h_t, c) -> phi -> (mu_t, Sigma_t) -> v_t
Decoder: Bi-LSTM(v_t, c) -> psi(y_t) -> z_tilde_t
"""

import tensorflow as tf
from tensorflow import keras
from keras.layers import (
    Input,
    Bidirectional,
    Concatenate,
    LSTM,
    Dense,
    MultiHeadAttention,
    LayerNormalization,
    Dropout,
    BatchNormalization,
)
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from .helper import Sampling


class PositionalEmbedding(keras.layers.Layer):
    """Learned positional embedding added to input sequences."""

    def __init__(self, max_seq_len: int, embed_dim: int, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.pos_embedding = self.add_weight(
            shape=(self.max_seq_len, self.embed_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="pos_embedding",
        )
        super(PositionalEmbedding, self).build(input_shape)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        pos_encoding = self.pos_embedding[:seq_len, :]
        pos_encoding = tf.expand_dims(pos_encoding, axis=0)
        return inputs + pos_encoding

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({"max_seq_len": self.max_seq_len, "embed_dim": self.embed_dim})
        return config


class VariationalMultiHeadAttention(keras.layers.Layer):
    """VMHA(h_t, c): Enriches Bi-LSTM hidden states h_t with condition c through
    self-attention (long-range temporal dependencies) and cross-attention
    (condition-aware representations). Preserves per-timestep temporal structure
    for subsequent variational sampling.
    """

    def __init__(
        self, num_heads: int, ff_dim: int, max_seq_len: int, rate: float = 0.3, **kwargs
    ):
        super(VariationalMultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len
        self.rate = rate

    def build(self, input_shape):
        h_shape, _ = input_shape
        h_dim = h_shape[-1]

        self.pos_embed = PositionalEmbedding(self.max_seq_len, h_dim)
        self.c_proj = Dense(
            h_dim, kernel_initializer="glorot_uniform", name="c_projection"
        )

        self.self_attn = MultiHeadAttention(
            num_heads=self.num_heads, key_dim=h_dim, name="self_attention"
        )
        self.cross_attn = MultiHeadAttention(
            num_heads=self.num_heads, key_dim=h_dim, name="cross_attention"
        )

        self.ffn = keras.Sequential(
            [
                Dense(
                    self.ff_dim, activation="relu", kernel_initializer="glorot_uniform"
                ),
                Dense(h_dim, kernel_initializer="glorot_uniform"),
            ],
            name="ffn",
        )

        self.ln_self = LayerNormalization(epsilon=1e-6, name="ln_self_attn")
        self.ln_cross = LayerNormalization(epsilon=1e-6, name="ln_cross_attn")
        self.ln_ffn = LayerNormalization(epsilon=1e-6, name="ln_ffn")

        self.drop_self = Dropout(self.rate, name="drop_self_attn")
        self.drop_cross = Dropout(self.rate, name="drop_cross_attn")
        self.drop_ffn = Dropout(self.rate, name="drop_ffn")

        super(VariationalMultiHeadAttention, self).build(input_shape)

    def call(self, inputs, training=False):
        h, c = inputs

        c_proj = self.c_proj(c)
        h = self.pos_embed(h)

        self_out = self.self_attn(h, h)
        self_out = self.drop_self(self_out, training=training)
        h = self.ln_self(h + self_out)

        cross_out = self.cross_attn(query=h, key=c_proj, value=c_proj)
        cross_out = self.drop_cross(cross_out, training=training)
        h = self.ln_cross(h + cross_out)

        ffn_out = self.ffn(h)
        ffn_out = self.drop_ffn(ffn_out, training=training)
        return self.ln_ffn(h + ffn_out)

    def get_config(self):
        config = super(VariationalMultiHeadAttention, self).get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "max_seq_len": self.max_seq_len,
                "rate": self.rate,
            }
        )
        return config


class CVAE(keras.Model):
    def __init__(
        self,
        encoder: keras.Model,
        decoder: keras.Model,
        window_size: int,
        kl_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super(CVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.window_size = window_size
        self.kl_weight = kl_weight

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, inputs, training=None, mask=None):
        dynamic_features, inp_data = inputs
        z_mean, z_log_var, z = self.encoder([dynamic_features, inp_data])
        pred = self.decoder([z, dynamic_features])
        return pred

    @property
    def metrics(self) -> list:
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data: list) -> dict:
        dynamic_features, inp_data = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([dynamic_features, inp_data])
            pred = self.decoder([z, dynamic_features])

            reconstruction_loss = K.mean(K.square(inp_data - pred))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=[1, 2]))
            total_loss = reconstruction_loss + self.kl_weight * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def get_CVAE(
    window_size: int,
    n_main_features: int,
    n_dyn_features: int,
    latent_dim: int,
    num_heads: int = 8,
    ff_dim: int = 256,
) -> tuple[keras.Model, keras.Model]:
    # --- Encoder ---
    inp = Input(shape=(window_size, n_main_features), name="input_main_CVAE")
    dyn_inp = Input(shape=(window_size, n_dyn_features), name="input_dyn_CVAE")

    # Bi-LSTM captures short-term temporal dependencies from z
    h = Bidirectional(
        LSTM(
            n_main_features,
            kernel_initializer="glorot_uniform",
            dropout=0.3,
            kernel_regularizer=l2(0.001),
            return_sequences=True,
            name="bidirectional_lstm_CVAE",
        ),
        merge_mode="ave",
    )(inp)

    h = BatchNormalization(name="batch_norm_encoder")(h)

    # VMHA(h_t, c): self-attention + cross-attention with condition
    enc = VariationalMultiHeadAttention(
        num_heads=num_heads,
        ff_dim=ff_dim,
        max_seq_len=window_size,
        name="vmha_CVAE",
    )([h, dyn_inp])

    enc = Dropout(0.3, name="dropout_CVAE")(enc)

    # phi: maps VMHA output to latent distribution parameters per timestep
    enc = Dense(
        latent_dim,
        activation="relu",
        kernel_regularizer=l2(0.001),
        kernel_initializer="glorot_uniform",
        name="dense_latent_CVAE",
    )(enc)

    enc = BatchNormalization(name="batch_norm_latent")(enc)

    z_mean = Dense(latent_dim, kernel_initializer="glorot_uniform", name="z_mean")(enc)
    z_log_var = Dense(
        latent_dim, kernel_initializer="glorot_uniform", name="z_log_var"
    )(enc)

    # Per-timestep sampling: v_t ~ N(mu_t, Sigma_t)
    z = Sampling(name="sampling_layer")([z_mean, z_log_var])

    encoder = Model([dyn_inp, inp], [z_mean, z_log_var, z], name="encoder_CVAE")

    # --- Decoder ---
    inp_z = Input(shape=(window_size, latent_dim), name="input_latent_CVAE")
    dec_dyn_inp = Input(
        shape=(window_size, n_dyn_features), name="input_dyn_decoder_CVAE"
    )

    # Concatenate latent sequence v_t with condition c_t
    dec = Concatenate(name="concat_decoder_inputs_CVAE")([inp_z, dec_dyn_inp])

    # y_tilde_t = Bi-LSTM(v_t, h_tilde_{t-1}, c)
    dec = Bidirectional(
        LSTM(
            n_main_features,
            kernel_initializer="glorot_uniform",
            return_sequences=True,
            dropout=0.3,
            kernel_regularizer=l2(0.001),
            name="bidirectional_lstm_decoder_CVAE",
        ),
        merge_mode="ave",
    )(dec)

    # psi: per-timestep transformation z_tilde_t = psi(y_tilde_t)
    out = Dense(
        n_main_features,
        kernel_regularizer=l2(0.001),
        kernel_initializer="glorot_uniform",
        name="dense_output_CVAE",
    )(dec)

    decoder = Model([inp_z, dec_dyn_inp], out, name="decoder_CVAE")

    return encoder, decoder
