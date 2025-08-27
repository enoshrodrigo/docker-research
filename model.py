# model_train_refactor.py
import os
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Input, concatenate
from tensorflow.keras.models import Model

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optional experiment tracking (kept like original)
import mlflow
import mlflow.tensorflow
try:
    from azureml.core import Run
    run = Run.get_context()
except Exception:
    run = None

# ---------------------------
# Reproducibility / constants
# ---------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Create a small function to check GPUs in modern TF
def gpu_available():
    gpus = tf.config.list_physical_devices("GPU")
    return len(gpus) > 0

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {gpu_available()}")

mlflow.autolog()  # optional: auto-logging for Keras/TensorFlow
# ---------------------------
# Model architecture builder
# ---------------------------
def build_advanced_model(input_shape, output_length=5):
    sequence_input = Input(shape=input_shape, name="sequence_input")

    # Parallel 1D convs for multi-scale feature extraction
    conv1 = Conv1D(64, kernel_size=3, padding='same', activation='relu')(sequence_input)
    conv2 = Conv1D(64, kernel_size=5, padding='same', activation='relu')(sequence_input)
    conv3 = Conv1D(64, kernel_size=7, padding='same', activation='relu')(sequence_input)

    conv_concat = concatenate([conv1, conv2, conv3])

    lstm1 = LSTM(128, return_sequences=True)(conv_concat)
    lstm1_dropout = Dropout(0.3)(lstm1)

    # Self-attention (Keras Attention)
    attention = tf.keras.layers.Attention()([lstm1_dropout, lstm1_dropout])
    enhanced_features = concatenate([lstm1_dropout, attention])

    lstm_final = LSTM(96, return_sequences=False)(enhanced_features)
    lstm_final_dropout = Dropout(0.3)(lstm_final)

    dense1 = Dense(64, activation='relu')(lstm_final_dropout)
    dense2 = Dense(32, activation='relu')(dense1)
    output = Dense(output_length, activation='linear')(dense2)

    model = Model(inputs=sequence_input, outputs=output)
    return model

# ---------------------------
# Loss: asymmetric Huber
# ---------------------------
def asymmetric_huber_loss(y_true, y_pred):
    delta = 1.0
    error = y_true - y_pred

    is_high_precip = tf.cast(y_true > 10.0, tf.float32)
    is_underprediction = tf.cast(error > 0, tf.float32)
    is_small_error = tf.cast(tf.abs(error) <= delta, tf.float32)

    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * (tf.abs(error) - 0.5 * delta)
    base_loss = is_small_error * squared_loss + (1 - is_small_error) * linear_loss

    weight = 1.0 + is_underprediction * (1.0 + is_high_precip * 2.0)
    return tf.reduce_mean(weight * base_loss)

# ---------------------------
# Data utilities
# ---------------------------
def extract_extreme_events(data, target_col='precipitation_sum', threshold=10, lead=14, lag=5):
    """Return merged (start, end) intervals around indices where target > threshold."""
    extreme_periods = []
    arr = data[target_col].values
    idxs = np.where(arr > threshold)[0]
    for idx in idxs:
        start = max(0, idx - lead)
        end = min(len(data), idx + lag)
        extreme_periods.append((start, end))
    if not extreme_periods:
        return []
    extreme_periods.sort()
    merged = [extreme_periods[0]]
    for cur in extreme_periods[1:]:
        prev = merged[-1]
        if cur[0] <= prev[1]:
            merged[-1] = (prev[0], max(prev[1], cur[1]))
        else:
            merged.append(cur)
    return merged

def create_enhanced_sequences(data, feature_cols, target_col='precipitation_sum',
                              seq_length=14, forecast_days=5, augment_times=3,
                              augmentation_std=0.05, augment_threshold=10):
    """
    Build X, y arrays and return also seq_start_indices (index of first day of each input sequence
    relative to the original dataframe). This helps map sequences back to dates.
    """
    features = data[feature_cols].values
    target = data[target_col].values
    X, y, starts = [], [], []

    n = len(data)
    # Regular sliding windows
    for i in range(n - seq_length - forecast_days + 1):
        X.append(features[i:i+seq_length].astype(np.float32))
        y.append(target[i+seq_length:i+seq_length+forecast_days].astype(np.float32))
        starts.append(i)

    # Augmentation for extreme periods
    extreme_periods = extract_extreme_events(data, target_col, threshold=augment_threshold,
                                             lead=seq_length, lag=forecast_days)
    for (start, end) in extreme_periods:
        # we will create augmented sequences whose input windows end before or within the extreme window
        # ensure we don't go beyond valid indices
        max_start_i = max(start, end - seq_length - forecast_days + 1)
        # iterate candidate start positions
        for _ in range(augment_times):
            for i in range(start, max_start_i):
                seq_features = features[i:i+seq_length].copy().astype(np.float32)
                seq_target = target[i+seq_length:i+seq_length+forecast_days].astype(np.float32)
                # add gaussian noise to features (small)
                noise = np.random.normal(loc=0.0, scale=augmentation_std, size=seq_features.shape).astype(np.float32)
                seq_features += noise
                X.append(seq_features)
                y.append(seq_target)
                starts.append(i)
    X = np.array(X)
    y = np.array(y)
    starts = np.array(starts, dtype=np.int32)
    return X, y, starts

# ---------------------------
# Plot / evaluation helpers
# ---------------------------
def save_plot(fig, filename):
    filepath = OUTPUT_DIR / filename
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    # If running under Azure ML, log
    if run is not None:
        try:
            run.log_image(name=filename, plot=fig)
        except Exception:
            pass

def generate_evaluation_plots(y_true, y_pred, sample_dates=None, num_samples=5):
    # choose samples with heavy rainfall to highlight if possible
    heavy_idx = [i for i, arr in enumerate(y_true) if arr.max() > 15]
    rng = np.random.RandomState(SEED)
    if len(heavy_idx) >= num_samples:
        indices = rng.choice(heavy_idx, size=num_samples, replace=False)
    else:
        normal_idx = [i for i in range(len(y_true)) if i not in heavy_idx]
        more_needed = num_samples - len(heavy_idx)
        pick_normal = rng.choice(normal_idx, size=more_needed, replace=False) if normal_idx else []
        indices = heavy_idx + list(pick_normal)

    days = np.arange(1, y_true.shape[1] + 1)

    fig, axs = plt.subplots(len(indices), 1, figsize=(12, 3 * len(indices)))
    if len(indices) == 1:
        axs = [axs]
    for ax, idx in zip(axs, indices):
        ax.plot(days, y_true[idx], marker='o', label='Actual')
        ax.plot(days, y_pred[idx], marker='x', linestyle='--', label='Predicted')
        ax.set_xlabel('Forecast day')
        ax.set_ylabel('Precipitation (mm)')
        title = f'Sample {idx}'
        if sample_dates is not None:
            title += f' - start {sample_dates[idx].strftime("%Y-%m-%d")}'
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
    save_plot(fig, 'negombo_prediction_samples.png')

    # Error histogram
    errors = (y_pred.flatten() - y_true.flatten())
    fig = plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50)
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel('Prediction error (predicted - actual)')
    plt.ylabel('Frequency')
    plt.title('Forecast error distribution')
    save_plot(fig, 'negombo_error_distribution.png')

    # Scatter actual vs predicted
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(y_true.flatten(), y_pred.flatten(), alpha=0.5, s=20)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--')  # 45-degree
    corr = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    plt.xlabel('Actual precipitation (mm)')
    plt.ylabel('Predicted precipitation (mm)')
    plt.title(f'Actual vs Predicted (corr={corr:.3f})')
    save_plot(fig, 'negombo_actual_vs_predicted.png')

# ---------------------------
# Main training function
# ---------------------------
def train_model(data_path='weather_data.csv',
                seq_length=14, forecast_days=5,
                test_fraction=0.2, augment_times=3):
    print("Loading and preprocessing data...")
    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    # Basic time-based features
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

    # Monsoon indicators (binary)
    df['is_southwest_monsoon'] = ((df['month'] >= 5) & (df['month'] <= 9)).astype(int)
    df['is_northeast_monsoon'] = ((df['month'] >= 12) | (df['month'] <= 2)).astype(int)
    df['is_inter_monsoon1'] = ((df['month'] >= 3) & (df['month'] <= 4)).astype(int)
    df['is_inter_monsoon2'] = ((df['month'] >= 10) & (df['month'] <= 11)).astype(int)

    # Derived features
    df['temp_range'] = df['temperature_2m_max'] - df['temperature_2m_min']
    df['humidity_proxy'] = df['apparent_temperature_mean'] - df['temperature_2m_mean']
    # 3-day moving average per city (if multiple cities exist)
    df['precip_ma3'] = df.groupby('city')['precipitation_sum'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    df['heavy_rainfall'] = (df['precipitation_sum'] > 10).astype(int)

    # Filter for a single city (as before). The name is not hard-coded in metadata.
    city_name = 'Negombo'
    city_df = df[df['city'] == city_name].sort_values('time').reset_index(drop=True)
    if city_df.empty:
        raise ValueError(f"No data found for city '{city_name}'")
    print(f"Selected city: {city_name} ({len(city_df)} records)")

    # Feature columns (order is important — save in metadata)
    feature_cols = [
        'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
        'apparent_temperature_max', 'apparent_temperature_min', 'apparent_temperature_mean',
        'shortwave_radiation_sum', 'precipitation_sum', 'rain_sum',
        'windspeed_10m_max', 'windgusts_10m_max', 'winddirection_10m_dominant',
        'month_sin', 'month_cos', 'day_sin', 'day_cos',
        'is_southwest_monsoon', 'is_northeast_monsoon',
        'is_inter_monsoon1', 'is_inter_monsoon2',
        'temp_range', 'humidity_proxy', 'precip_ma3'
    ]

    # Ensure all feature columns are present
    missing = [c for c in feature_cols if c not in city_df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    # Create sequences (X,y) and start indices so we can map back to dates
    X, y, starts = create_enhanced_sequences(city_df, feature_cols, target_col='precipitation_sum',
                                            seq_length=seq_length, forecast_days=forecast_days,
                                            augment_times=augment_times)
    print(f"Total sequences (including augment): {len(X)}")

    # Train / test split by chronological order to avoid leakage
    n_samples = len(X)
    test_size = int(n_samples * test_fraction)
    train_size = n_samples - test_size

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    starts_train, starts_test = starts[:train_size], starts[train_size:]

    # Flatten and scale features using only training statistics (no leakage)
    X_train_shape = X_train.shape
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_train = X_train_flat.reshape(X_train_shape)

    # Transform test with the same scaler
    X_test_shape = X_test.shape
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    X_test_flat = scaler.transform(X_test_flat)
    X_test = X_test_flat.reshape(X_test_shape)

    # Save scaler and metadata for server-side evaluation (prevent leakage)
    joblib.dump(scaler, OUTPUT_DIR / "scaler.joblib")
    metadata = {
        "feature_cols": feature_cols,
        "seq_length": seq_length,
        "forecast_days": forecast_days,
        "city": city_name,
        "created_at": datetime.utcnow().isoformat(),
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Build model
    print("Building model...")
    model = build_advanced_model(input_shape=X_train.shape[1:], output_length=forecast_days)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=asymmetric_huber_loss,
                  metrics=['mae'])

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    ]

    # Training
    print("Starting training...")
    history = model.fit(X_train, y_train,
                        epochs=100,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=callbacks,
                        verbose=2)

    # Save model and history
    model_save_path = OUTPUT_DIR / "negombo_model.keras"
    model.save(model_save_path)
    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)

    # Predictions & evaluation on test set
    print("Evaluating on test set...")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())
    rmse = np.sqrt(mean_squared_error(y_test.flatten(), y_pred.flatten()))
    r2 = r2_score(y_test.flatten(), y_pred.flatten())
    corr = np.corrcoef(y_test.flatten(), y_pred.flatten())[0, 1]
    bias = float(np.mean(y_pred.flatten() - y_test.flatten()))

    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, Corr: {corr:.4f}, Bias: {bias:.4f}")

    # Log to Azure ML if available
    if run is not None:
        run.log('mae', float(mae))
        run.log('rmse', float(rmse))
        run.log('r2', float(r2))
        run.log('correlation', float(corr))
        run.log('bias', float(bias))

    # Save evaluation metrics
    eval_summary = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "correlation": float(corr),
        "bias": float(bias),
        "n_test_samples": int(len(X_test)),
    }
    with open(OUTPUT_DIR / "eval_summary.json", "w") as f:
        json.dump(eval_summary, f, indent=2)

    # Save worst errors for manual inspection (map back to dates)
    errors = (y_pred.flatten() - y_test.flatten())
    abs_errors = np.abs(errors)
    worst_idx_flat = np.argsort(-abs_errors)[:20]  # top 20 worst absolute errors (flattened)
    rows = []
    # We need to map flattened index back to (sample_idx, day_idx)
    n_days = y_test.shape[1]
    for flat_idx in worst_idx_flat:
        sample_idx = flat_idx // n_days
        day_idx = flat_idx % n_days
        seq_start = starts_test[sample_idx]  # start index in original df for that test sample
        # the prediction corresponds to date = df.time[seq_start + seq_length + day_idx]
        actual_date = city_df['time'].iloc[seq_start + seq_length + day_idx]
        rows.append({
            "sample_index": int(sample_idx),
            "forecast_day": int(day_idx + 1),
            "date": str(actual_date.date()),
            "predicted": float(y_pred.flatten()[flat_idx]),
            "actual": float(y_test.flatten()[flat_idx]),
            "error": float(errors[flat_idx])
        })
    worst_df = pd.DataFrame(rows)
    worst_df.to_csv(OUTPUT_DIR / "worst_errors.csv", index=False)
    print(f"Saved worst errors to {OUTPUT_DIR / 'worst_errors.csv'}")

    # Generate and save plots (pass sample_dates for context)
    # Build sample_dates mapping: for each test sample i, its input sequence start date
    test_sample_dates = [city_df['time'].iloc[s + seq_length] for s in starts_test]
    generate_evaluation_plots(y_test, y_pred, sample_dates=test_sample_dates, num_samples=5)

    print("Training complete.")
    return {
        "model_path": str(model_save_path),
        "scaler_path": str(OUTPUT_DIR / "scaler.joblib"),
        "metadata_path": str(OUTPUT_DIR / "metadata.json"),
        "eval_summary": eval_summary,
        "worst_errors_csv": str(OUTPUT_DIR / "worst_errors.csv")
    }

# ---------------------------
# Run training when invoked
# ---------------------------
if __name__ == "__main__":
    results = train_model(data_path='weather_data.csv',
                          seq_length=14,
                          forecast_days=5,
                          test_fraction=0.2,
                          augment_times=3)
    print(json.dumps(results, indent=2))
