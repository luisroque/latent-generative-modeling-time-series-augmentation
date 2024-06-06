from lgta.model.create_dataset_versions_vae import (
    CreateTransformedVersionsCVAE,
)
import os
from tensorflow import keras
from keras.callbacks import EarlyStopping

from lgta.model.models import CVAE, get_CVAE


dataset = "tourism"
freq = "M"
weights_file = "model_weights/tourism_vae_weights.h5"

create_dataset_vae = CreateTransformedVersionsCVAE(
    dataset_name=dataset, freq=freq, dynamic_feat_trig=False
)

# Verify if weights file is created
if os.path.exists(weights_file):
    print("Weights file created successfully.")

# Load the weights after training
create_dataset_vae.features_input = create_dataset_vae._feature_engineering(
    create_dataset_vae.n_train
)
try:
    encoder, decoder = get_CVAE(
        static_features=create_dataset_vae.features_input[2],
        dynamic_features=create_dataset_vae.features_input[0],
        window_size=create_dataset_vae.window_size,
        n_features=create_dataset_vae.n_features,
        n_features_concat=create_dataset_vae.n_features_concat,
        latent_dim=2,
        embedding_dim=8,
    )
    learning_rate = (0.001,)

    cvae = CVAE(encoder, decoder, create_dataset_vae.window_size)
    cvae.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

    es = EarlyStopping(
        patience=30,
        verbose=1,
        monitor="loss",
        mode="auto",
        restore_best_weights=True,
    )

    weights_folder = "model_weights"
    os.makedirs(weights_folder, exist_ok=True)

    weights_file = os.path.join(
        weights_folder, f"{create_dataset_vae.dataset_name}_vae_weights.h5"
    )
    history = None

    _ = cvae(create_dataset_vae.features_input)
    print("Loading existing weights...")
    cvae.load_weights(weights_file)
    print("Weights loaded successfully.")
except Exception as e:
    print(f"Error loading weights: {e}")

# Display model summary
print("Model summary:")
create_dataset_vae.cvae.summary()

for layer in create_dataset_vae.cvae.layers:
    print(f"Layer: {layer.name}, Weights: {len(layer.get_weights())}")
