import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from lgta.model.create_dataset_versions_vae import CreateTransformedVersionsCVAE
from lgta.visualization.model_visualization import plot_loss
from lgta.feature_engineering.feature_transformations import detemporalize


# Define dataset details
dataset = "tourism"
freq = "M"
top = None

# Step 1: Initialize and Train the Model
create_dataset_vae = CreateTransformedVersionsCVAE(
    dataset_name=dataset, freq=freq, top=top, dynamic_feat_trig=True
)
model, history, _ = create_dataset_vae.fit()
# plot_loss(history)

# Feature engineering and stacking dynamic features for prediction
(dynamic_feat, X_inp, static_feat), _ = create_dataset_vae._feature_engineering(
    create_dataset_vae.n, val_steps=0
)
stacked_dynamic_feat = tf.stack(dynamic_feat, axis=-1)


# Step 1: Plot Each Z Mean Component as a Time Series
def plot_z_means_over_time(z_means):
    """
    Plot each Z mean component as a time series.

    Args:
        z_means (np.array): Array of z mean values over time (e.g., shape (time, latent_dim))
    """
    n_dims = z_means.shape[1]
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, n_dims * 2))
    components = ["trend", "seasonality", "level"]
    for i, component in zip(range(n_dims), components):
        axes[i].plot(z_means[:, i], color="blue", label=f"Z mean {i+1}")
        axes[i].set_title(f"Z Mean {component}")
        axes[i].legend()
    plt.tight_layout()
    plt.show()


# Extract Z mean and plot over time
z_means, z_log_vars, _ = model.encoder.predict([X_inp, stacked_dynamic_feat])
plot_z_means_over_time(z_means)


# Step 2: Plot Original vs. Modified Latent Representations
def plot_original_vs_modified_z(z_original, z_modified, title):
    """
    Plot the original and modified Z mean components for comparison.

    Args:
        z_original (np.array): Original Z mean values
        z_modified (np.array): Modified Z mean values
        title (str): Title for the plot
    """
    n_dims = z_original.shape[1]
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, n_dims * 2))
    for i in range(n_dims):
        axes[i].plot(z_original[:, i], color="blue", label="Original Z")
        axes[i].plot(z_modified[:, i], color="red", linestyle="--", label="Modified Z")
        axes[i].set_title(f"Z Mean {i+1}")
        axes[i].legend()
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# Define indices for trend, seasonality, and level based on updated latent structure
trend_indices = [0]  # Trend mean
seasonality_indices = [1]  # Seasonality mean
level_indices = [2]  # Level mean


# Step 3: Manipulate Latent Components and Generate Synthetic Data
def manipulate_and_generate(z, stacked_dynamic_feat, component_indices, offset=0.5):
    z_modified = np.copy(z)
    z_modified[:, component_indices] += offset  # Modify specific latent components

    # Generate synthetic data with modified latent space
    generated_data = model.decoder.predict([z_modified, stacked_dynamic_feat])
    return generated_data


# Generate original and modified synthetic data
preds_original = model.decoder.predict([z_means, stacked_dynamic_feat])
generated_data_trend = detemporalize(
    manipulate_and_generate(z_means, stacked_dynamic_feat, trend_indices, offset=0.5),
    12,
)
generated_data_seasonality = detemporalize(
    manipulate_and_generate(
        z_means, stacked_dynamic_feat, seasonality_indices, offset=0.5
    ),
    12,
)
generated_data_level = detemporalize(
    manipulate_and_generate(z_means, stacked_dynamic_feat, level_indices, offset=0.5),
    12,
)


# Step 4: Visualize Original vs Modified Time Series
def plot_original_vs_modified(X_orig, generated_data, title):
    plt.figure(figsize=(10, 6))
    plt.plot(X_orig[:, 0], label="Original", color="black")
    plt.plot(generated_data[:, 0], label="Modified", color="red")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.show()


# Load original time series data for comparison
X_orig = create_dataset_vae.X_train_raw  # Raw data before scaling if available


# Updated inverse_transform function to handle 3D arrays
def inverse_transform(data, scaler):
    if not scaler:
        return data
    # Reshape from (samples, timesteps, features) to (samples*timesteps, features)
    original_shape = data.shape
    data_reshaped = data.reshape(-1, original_shape[-1])
    # Apply inverse scaling
    data_inverse = scaler.inverse_transform(data_reshaped)
    # Reshape back to original 3D shape
    return data_inverse.reshape(original_shape)


# Apply inverse_transform to all generated datasets
generated_data_trend = inverse_transform(
    generated_data_trend, create_dataset_vae.scaler_target
)
generated_data_seasonality = inverse_transform(
    generated_data_seasonality, create_dataset_vae.scaler_target
)
generated_data_level = inverse_transform(
    generated_data_level, create_dataset_vae.scaler_target
)

# Plot Original vs. Modified series with inverse-transformed values
plot_original_vs_modified(
    X_orig, generated_data_trend, "Original vs Trend-Modified (Inversed Scale)"
)
plot_original_vs_modified(
    X_orig,
    generated_data_seasonality,
    "Original vs Seasonality-Modified (Inversed Scale)",
)
plot_original_vs_modified(
    X_orig, generated_data_level, "Original vs Level-Modified (Inversed Scale)"
)
