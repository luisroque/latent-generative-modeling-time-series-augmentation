import tensorflow as tf
import pandas as pd
from tensorflow import keras
from keras.layers import (
    Input,
    Bidirectional,
    Concatenate,
    LSTM,
    Reshape,
    Flatten,
    Dense,
    MultiHeadAttention,
    Attention,
    LayerNormalization,
    GlobalAveragePooling1D,
)
from keras.regularizers import l2
from keras.layers import Dropout, RepeatVector, Embedding
from keras import backend as K
import numpy as np
from .helper import Sampling
from keras.models import Model


def get_positional_encoding(max_seq_len, embed_dim):
    """Generates sinusoidal positional encodings."""
    positional_encoding = np.array(
        [
            (
                [
                    pos / np.power(10000, 2 * (i // 2) / embed_dim)
                    for i in range(embed_dim)
                ]
                if pos != 0
                else np.zeros(embed_dim)
            )
            for pos in range(max_seq_len)
        ]
    )
    positional_encoding[1:, 0::2] = np.sin(positional_encoding[1:, 0::2])  # dim 2i
    positional_encoding[1:, 1::2] = np.cos(positional_encoding[1:, 1::2])  # dim 2i+1
    return tf.cast(positional_encoding, dtype=tf.float32)


class TransformerBlockWithPosEncoding(tf.keras.layers.Layer):
    def __init__(self, num_heads, ff_dim, max_seq_len, rate=0.1, **kwargs):
        super(TransformerBlockWithPosEncoding, self).__init__()
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len
        self.rate = rate
        self.att = None
        self.ffn = None
        self.layernorm1 = None
        self.layernorm2 = None
        self.dropout1 = None
        self.dropout2 = None

    def build(self, input_shape):
        embed_dim = input_shape[-1]
        self.pos_encoding = get_positional_encoding(self.max_seq_len, embed_dim)[
            tf.newaxis, ...
        ]
        self.mhatt = MultiHeadAttention(
            num_heads=self.num_heads, key_dim=embed_dim, name="multi_head_attention"
        )
        self.ffn = tf.keras.Sequential(
            [
                Dense(self.ff_dim, activation="relu", name="dense_relu"),
                Dense(embed_dim, name="dense_output"),
            ],
            name="ffn",
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6, name="layer_norm_1")
        self.layernorm2 = LayerNormalization(epsilon=1e-6, name="layer_norm_2")
        self.dropout1 = Dropout(self.rate, name="dropout_1")
        self.dropout2 = Dropout(self.rate, name="dropout_2")
        super(TransformerBlockWithPosEncoding, self).build(input_shape)

    def call(self, inputs, training=False):
        seq_len = tf.shape(inputs)[1]
        embed_dim = tf.shape(inputs)[-1]
        pos_encoding = self.pos_encoding[:, :seq_len, :embed_dim]
        inputs += pos_encoding
        attn_output = self.mhatt(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        ffn_output = self.layernorm2(out1 + ffn_output)
        context_vector = tf.reduce_mean(ffn_output, axis=1)

        return context_vector

    def get_config(self):
        config = super(TransformerBlockWithPosEncoding, self).get_config()
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
    """
    Conditional Variational Autoencoder class that implements a custom architecture of encoder and decoder
    that handles raw data plus dynamic and static features as well as custom metrics to track

    Attributes
    ----------
    encoder : keras.Model
        encoder model
    decoder : keras.Model
        decoder model
    window_size : int
        rolling window
    reconstruction_loss_tracker : keras.metrics
        loss computing mean square error on the reconstruction and original data
    kl_loss_tracker: keras.metrics
        kl divergency between simpler learned distribution and actual distribution
    total_loss_tracker : keras.metrics
        sum of reconstruction and kl loss

    """

    def __init__(
        self, encoder: keras.Model, decoder: keras.Model, window_size: int, **kwargs
    ) -> None:
        super(CVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.window_size = window_size
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, inputs, training=None, mask=None):
        dynamic_features, inp_data, static_features = inputs

        z_mean, z_log_var, z = self.encoder(
            dynamic_features + inp_data + static_features
        )
        pred = self.decoder([z] + dynamic_features + static_features)
        return pred

    @property
    def metrics(self) -> list:
        """
        Metrics to track for the VAE

        :return: metrics to track
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data: list) -> dict:
        """
        Custom training procedure designed to train a VAE and report the relevant metrics

        :param data: input data to the model
        :param window_size: rolling window

        :return: metrics
        """
        dynamic_features, inp_data, static_features = data[0]
        dynamic_features = list(dynamic_features)
        inp_data = list(inp_data)
        static_features = list(static_features)

        device = "/gpu:0" if tf.config.list_physical_devices("GPU") else "/cpu:0"

        with tf.device(device):
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(
                    dynamic_features + inp_data + static_features
                )
                pred = self.decoder([z] + dynamic_features + static_features)

                reconstruction_loss = (
                    K.mean(K.square(inp_data - pred)) * self.window_size
                )
                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                total_loss = reconstruction_loss + kl_loss

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


def get_flatten_size_encoder(
    static_features: dict,
    dynamic_features_df: pd.DataFrame,
    window_size: int,
    n_features: int,
    n_features_concat: int,
    num_heads: int = 8,
    ff_dim: int = 256,
    max_seq_len: int = 10,
) -> int:
    inp = Input(shape=(window_size, n_features), name="input_main")

    dynamic_features_inp = []
    for feature in range(len(dynamic_features_df.columns)):
        inp_dyn_feat = Input(
            shape=(window_size, 1), name=f"input_dynamic_feature_{feature}"
        )
        dynamic_features_inp.append(inp_dyn_feat)

    static_feat_inp = []
    for feature, arr in static_features.items():
        inp_static_feat = Input(
            shape=(n_features, 1), name=f"input_static_feature_{feature}"
        )
        static_feat_inp.append(inp_static_feat)

    enc = Concatenate(name="concat_encoder_inputs")(dynamic_features_inp + [inp])
    enc = TransformerBlockWithPosEncoding(
        num_heads, ff_dim, max_seq_len, name="transformer_block_encoder"
    )(enc)
    enc = Bidirectional(
        LSTM(
            n_features,
            kernel_initializer="random_uniform",
            input_shape=(window_size, n_features_concat),
        ),
        merge_mode="ave",
        name="bidirectional_lstm_encoder",
    )(enc)

    enc = Reshape((-1, 1), name="reshape_encoder_output")(enc)
    enc = Concatenate(name="concat_encoder_final")([enc] + static_feat_inp)
    enc = Flatten(name="flatten_encoder_output")(enc)

    temp_model = Model(
        dynamic_features_inp + [inp] + static_feat_inp, enc, name="encoder_temp_model"
    )
    flatten_size = temp_model.output_shape[1]
    return flatten_size


class PrintShapeLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(PrintShapeLayer, self).__init__(name=name)

    def call(self, inputs):
        tf.print(f"{self.name} input shape:", tf.shape(inputs))
        return inputs


def get_CVAE(
    static_features: list,
    dynamic_features: list,
    window_size: int,
    n_features: int,
    n_features_concat: int,
    latent_dim: int,
    embedding_dim: int,
    num_heads: int = 8,
    ff_dim: int = 256,
    max_seq_len: int = 10,
) -> tuple[keras.Model, keras.Model]:
    inp = Input(shape=(window_size, n_features), name="input_main_CVAE")

    dynamic_features_inp = []
    dynamic_features_emb = []
    for feature in range(len(dynamic_features)):
        inp_dyn_feat = Input(
            shape=(window_size,), name=f"input_dynamic_feature_{feature}_CVAE"
        )
        dynamic_features_inp.append(inp_dyn_feat)
        emb_dyn_feat = Embedding(
            input_dim=int(dynamic_features[feature].max() + 1),
            output_dim=embedding_dim,
            name=f"embedding_dynamic_feature_{feature}_CVAE",
        )(inp_dyn_feat)
        emb_dyn_feat = Reshape(
            (window_size, embedding_dim), name=f"reshape_dynamic_feature_{feature}_CVAE"
        )(emb_dyn_feat)
        dynamic_features_emb.append(emb_dyn_feat)

    static_feat_inp = []
    static_feat_emb = []
    for feature in range(len(static_features)):
        inp_static_feat = Input(
            shape=(n_features,), name=f"input_static_feature_{feature}_CVAE"
        )
        static_feat_inp.append(inp_static_feat)
        emb_static_feat = Embedding(
            input_dim=int(static_features[feature].max() + 1),
            output_dim=embedding_dim,
            name=f"embedding_static_feature_{feature}_CVAE",
        )(inp_static_feat)
        emb_static_feat = Flatten(name=f"flatten_static_feature_{feature}_CVAE")(
            emb_static_feat
        )
        static_feat_emb.append(emb_static_feat)

    enc = Concatenate(name="concat_encoder_inputs_CVAE")(dynamic_features_emb + [inp])

    print("enc_before_lstm.shape:", enc.shape)

    enc = Bidirectional(
        LSTM(
            n_features,
            kernel_initializer="random_uniform",
            input_shape=(window_size, n_features_concat),
            dropout=0.5,
            kernel_regularizer=l2(0.001),
            name="bidirectional_lstm_CVAE",
            return_sequences=True,
        ),
        merge_mode="ave",
    )(enc)

    print("enc_before_transf.shape:", enc.shape)

    enc = TransformerBlockWithPosEncoding(
        num_heads=num_heads,
        ff_dim=ff_dim,
        max_seq_len=window_size,
        name="transformer_block_CVAE",
    )(enc)

    print("enc_after_transf.shape:", enc.shape)

    enc = Dropout(0.5, name="dropout_CVAE")(enc)
    enc = Concatenate(name="concat_encoder_final_CVAE")([enc] + static_feat_emb)
    enc = Flatten(name="flatten_encoder_output_CVAE")(enc)
    enc = Dense(
        latent_dim,
        activation="relu",
        kernel_regularizer=l2(0.001),
        name="dense_latent_CVAE",
    )(enc)

    z_mean = Dense(latent_dim, name="z_mean")(enc)
    z_log_var = Dense(latent_dim, name="z_log_var")(enc)

    z = Sampling(name="sampling_layer")([z_mean, z_log_var])

    encoder = Model(
        dynamic_features_inp + [inp] + static_feat_inp,
        [z_mean, z_log_var, z],
        name="encoder_CVAE",
    )

    inp_z = Input(shape=(latent_dim,), name="input_latent_CVAE")

    dec = RepeatVector(window_size, name="repeat_vector_CVAE")(inp_z)
    dec = Reshape((window_size, -1), name="reshape_decoder_input_CVAE")(dec)
    dec = Concatenate(name="concat_decoder_inputs_CVAE")([dec] + dynamic_features_emb)

    dec = Bidirectional(
        LSTM(
            n_features,
            kernel_initializer="random_uniform",
            input_shape=(window_size, latent_dim),
            return_sequences=True,
            dropout=0.5,
            kernel_regularizer=l2(0.001),
            name="bidirectional_lstm_decoder_CVAE",
        ),
        merge_mode="ave",
    )(dec)

    out = Flatten(name="flatten_decoder_output_CVAE")(dec)
    out = Concatenate(name="concat_decoder_final_CVAE")([out] + static_feat_emb)
    out = Dense(
        window_size * n_features, kernel_regularizer=l2(0.001), name="dense_output_CVAE"
    )(out)
    out = Reshape((window_size, n_features), name="reshape_final_output_CVAE")(out)

    decoder = Model(
        [inp_z] + dynamic_features_inp + static_feat_inp, out, name="decoder_CVAE"
    )

    return encoder, decoder
