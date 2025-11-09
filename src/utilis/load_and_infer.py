import os
import re
import glob
import numpy as np
import mne
import tensorflow as tf
from scipy import signal
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers  # type: ignore
from tqdm import tqdm

# ============================================
# CONFIGURATION
# ============================================
# Channel labels (must match training)
ch_labels = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
]

# Preprocessing parameters (must match training)
TIME_WINDOW = 8  # seconds
TIME_STEP = 4  # seconds
TARGET_WINDOW_SEC = 3  # Final window size
BANDPASS_RANGE = (0.5, 30)  # Hz
NOTCH_FREQ = 50  # Hz (use 60 for US data)


# ============================================
# Step 1: Model Architecture (Same as training)
# ============================================
class PositionalEmbedding(layers.Layer):
    def __init__(self, maxlen, d_model, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.d_model = d_model
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]  # type: ignore
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_embeddings = self.pos_emb(positions)
        return x + pos_embeddings

    def get_config(self):
        config = super().get_config()
        config.update({"maxlen": self.maxlen, "d_model": self.d_model})
        return config


# ============================================
# Step 2: Load Model
# ============================================
def load_model_for_inference(save_dir):
    """Load the trained model from SavedModel format"""
    savedmodel_path = os.path.join(save_dir, "cnn_attention_savedmodel")

    if not os.path.exists(savedmodel_path):
        raise FileNotFoundError(f"SavedModel not found at: {savedmodel_path}")

    print(f"Loading model from: {savedmodel_path}")
    loaded = tf.saved_model.load(savedmodel_path)
    infer = loaded.signatures["serving_default"]  # type: ignore

    # Create wrapper for easier use
    class ModelWrapper:
        def __init__(self, infer_func):
            self.infer = infer_func

        def predict(self, x, verbose=0):
            if isinstance(x, np.ndarray):
                x = tf.convert_to_tensor(x, dtype=tf.float32)
            output = self.infer(x)
            result = list(output.values())[0]
            return result.numpy()

    model = ModelWrapper(infer)
    print("✓ Model loaded successfully!")
    return model


# ============================================
# Step 3: EEG Preprocessing Functions
# ============================================
def preprocess_eeg_window(window_data, original_fs, target_window_sec=3, bandpass_range=(0.5, 30), notch_freq=50):
    """
    Preprocess a single EEG window (n_channels, n_times)
    Returns: (n_windows, n_channels, target_samples)
    """
    n_channels, n_times = window_data.shape
    target_window_size = int(target_window_sec * original_fs)

    # Create filter coefficients
    nyquist = 0.5 * original_fs
    low, high = bandpass_range
    b, a = signal.butter(4, [low / nyquist, high / nyquist], btype="band")  # type: ignore
    notch_b, notch_a = signal.iirnotch(notch_freq, 30, original_fs)

    # Apply filters to each channel
    filtered_data = np.zeros_like(window_data)
    for ch in range(n_channels):
        # Bandpass filter
        channel_data = signal.filtfilt(b, a, window_data[ch])
        # Notch filter
        channel_data = signal.filtfilt(notch_b, notch_a, channel_data)
        filtered_data[ch] = channel_data

    # Z-score normalization per channel
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(filtered_data.T).T

    # Segment into 3-second windows
    n_windows = n_times // target_window_size
    processed_windows = []

    for w in range(n_windows):
        start = w * target_window_size
        end = start + target_window_size
        window = normalized_data[:, start:end]
        processed_windows.append(window)

    return np.stack(processed_windows) if processed_windows else np.array([])


def load_and_preprocess_edf(
    edf_file, ch_labels, time_window=8, time_step=4, target_window_sec=3, bandpass_range=(0.5, 30), notch_freq=50
):
    """
    Load an EDF file and preprocess it for inference

    Returns:
        X: Preprocessed data (n_samples, n_channels, n_timepoints)
        window_info: List of dicts with timing information for each window
    """
    print(f"\nProcessing: {os.path.basename(edf_file)}")

    # Load EDF file
    edf = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)

    # Check channel availability
    channel_matches = sum(any(re.match(c, l) for l in edf.ch_names) for c in ch_labels)

    if channel_matches != len(ch_labels):
        print(f"⚠ Warning: Only {channel_matches}/{len(ch_labels)} channels matched")
        edf.close()
        return None, None

    # Rename channels to match training
    ch_mapping = {next(l for l in edf.ch_names if re.match(c, l)): c for c in ch_labels}
    edf.rename_channels(ch_mapping)

    # Get data
    signals = edf.get_data(picks=ch_labels) * 1e6  # type: ignore # Convert to µV
    fs = int(edf.info["sfreq"])

    # Calculate windowing parameters
    step_window = time_window * fs
    step = time_step * fs
    n_windows = (signals.shape[1] - step_window) // step

    print(f"  Sampling rate: {fs} Hz")
    print(f"  Duration: {signals.shape[1]/fs:.1f} seconds")
    print(f"  Creating {n_windows} windows...")

    all_processed_windows = []
    window_info = []

    # Process each window
    for i in range(n_windows):
        start_sample = i * step
        end_sample = start_sample + step_window
        window_data = signals[:, start_sample:end_sample]

        # Preprocess this window
        processed = preprocess_eeg_window(window_data, fs, target_window_sec, bandpass_range, notch_freq)

        if len(processed) > 0:
            all_processed_windows.extend(processed)

            # Store timing info for each sub-window
            for j in range(len(processed)):
                window_info.append(
                    {
                        "file": os.path.basename(edf_file),
                        "window_idx": i,
                        "sub_window_idx": j,
                        "start_time": start_sample / fs,
                        "end_time": end_sample / fs,
                        "sub_start_time": (start_sample + j * target_window_sec * fs) / fs,
                        "sub_end_time": (start_sample + (j + 1) * target_window_sec * fs) / fs,
                    }
                )

    edf.close()

    if all_processed_windows:
        X = np.stack(all_processed_windows)
        print(f"  ✓ Created {len(X)} preprocessed samples")
        return X, window_info
    else:
        return None, None


# ============================================
# Step 4: Inference Functions
# ============================================
def predict_edf_file(model, edf_file, ch_labels, **preprocess_kwargs):
    """
    Run inference on a single EDF file

    Returns:
        predictions: Array of predicted classes (0=non-seizure, 1=seizure)
        probabilities: Array of seizure probabilities
        window_info: Timing information for each prediction
    """
    # Load and preprocess
    X, window_info = load_and_preprocess_edf(edf_file, ch_labels, **preprocess_kwargs)

    if X is None:
        print("  ✗ Failed to process file")
        return None, None, None

    # Transpose to match model input: (batch, time, channels)
    X = X.transpose(0, 2, 1)

    print(f"  Running inference on {len(X)} windows...")

    # Get predictions
    probabilities = model.predict(X, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    seizure_probs = probabilities[:, 1]  # Probability of seizure class

    # Add predictions to window info
    for i, info in enumerate(window_info):  # type: ignore
        info["prediction"] = predictions[i]
        info["seizure_probability"] = seizure_probs[i]

    n_seizures = np.sum(predictions == 1)
    print(f"  ✓ Detected {n_seizures} potential seizure windows ({n_seizures/len(predictions)*100:.1f}%)")

    return predictions, seizure_probs, window_info


def predict_patient_folder(model, patient_folder, ch_labels, **preprocess_kwargs):
    """
    Run inference on all EDF files in a patient folder

    Returns:
        all_results: List of dicts containing results for each file
    """
    edf_files = sorted(glob.glob(os.path.join(patient_folder, "*.edf")))

    if not edf_files:
        print(f"No EDF files found in {patient_folder}")
        return []

    print(f"\nProcessing {len(edf_files)} files from {os.path.basename(patient_folder)}")
    print("=" * 60)

    all_results = []

    for edf_file in tqdm(edf_files, desc="Processing files"):
        predictions, probs, window_info = predict_edf_file(model, edf_file, ch_labels, **preprocess_kwargs)

        if predictions is not None:
            all_results.append(
                {
                    "file": os.path.basename(edf_file),
                    "predictions": predictions,
                    "probabilities": probs,
                    "window_info": window_info,
                    "n_seizure_windows": np.sum(predictions == 1),
                    "total_windows": len(predictions),
                }
            )

    return all_results


# ============================================
# Step 5: Results Analysis and Export
# ============================================
def save_results(results, output_dir):
    """Save inference results to CSV files"""
    os.makedirs(output_dir, exist_ok=True)

    import pandas as pd

    # Detailed results per window
    all_windows = []
    for result in results:
        for info in result["window_info"]:
            all_windows.append(info)

    df_windows = pd.DataFrame(all_windows)
    windows_path = os.path.join(output_dir, "detailed_predictions.csv")
    df_windows.to_csv(windows_path, index=False)
    print(f"\n✓ Detailed predictions saved to: {windows_path}")

    # Summary per file
    summary_data = []
    for result in results:
        summary_data.append(
            {
                "file": result["file"],
                "total_windows": result["total_windows"],
                "seizure_windows": result["n_seizure_windows"],
                "seizure_percentage": result["n_seizure_windows"] / result["total_windows"] * 100,
                "max_seizure_prob": np.max(result["probabilities"]),
                "mean_seizure_prob": np.mean(result["probabilities"]),
            }
        )

    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "file_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"✓ Summary saved to: {summary_path}")

    return df_windows, df_summary, summary_data


# ============================================
# Step 6: Main Inference Pipeline
# ============================================
def run_inference(model_dir, data_path, output_dir="inference_results"):
    """
    Complete inference pipeline

    Args:
        model_dir: Path to saved model directory
        data_path: Path to patient folder (e.g., 'chb01') or single EDF file
        output_dir: Where to save results
    """
    # Load model
    model = load_model_for_inference(model_dir)

    # Check if data_path is a file or directory
    if os.path.isfile(data_path) and data_path.endswith(".edf"):
        # Single file inference
        predictions, probs, window_info = predict_edf_file(
            model,
            data_path,
            ch_labels,
            time_window=TIME_WINDOW,
            time_step=TIME_STEP,
            target_window_sec=TARGET_WINDOW_SEC,
            bandpass_range=BANDPASS_RANGE,
            notch_freq=NOTCH_FREQ,
        )

        if predictions is not None:
            results = [
                {
                    "file": os.path.basename(data_path),
                    "predictions": predictions,
                    "probabilities": probs,
                    "window_info": window_info,
                    "n_seizure_windows": np.sum(predictions == 1),
                    "total_windows": len(predictions),
                }
            ]
        else:
            results = []

    else:
        # Patient folder inference
        results = predict_patient_folder(
            model,
            data_path,
            ch_labels,
            time_window=TIME_WINDOW,
            time_step=TIME_STEP,
            target_window_sec=TARGET_WINDOW_SEC,
            bandpass_range=BANDPASS_RANGE,
            notch_freq=NOTCH_FREQ,
        )

    # Save results
    if results:
        df_windows, df_summary, summary_data = save_results(results, output_dir)

        print("\n" + "=" * 60)
        print("INFERENCE COMPLETE!")
        print("=" * 60)
        print(f"Total files processed: {len(results)}")
        print(f"Total windows analyzed: {sum(r['total_windows'] for r in results)}")
        print(f"Seizure windows detected: {sum(r['n_seizure_windows'] for r in results)}")

        return results, df_windows, df_summary, summary_data
    else:
        print("\nNo results generated")
        return None, None, None, None


# ============================================
# EXAMPLE USAGE
# ============================================
if __name__ == "__main__":
    # Update these paths
    MODEL_DIR = "extracted_models/saved_models"

    # Option 1: Single file
    # DATA_PATH = 'path/to/chb01/chb01_01.edf'

    # Option 2: Patient folder
    DATA_PATH = "data/chb01_03.edf"

    # Option 3: Multiple patients
    # for patient in ['chb01', 'chb02', 'chb03']:
    #     DATA_PATH = f'path/to/{patient}'
    #     run_inference(MODEL_DIR, DATA_PATH, output_dir=f'results_{patient}')

    # Run inference
    results, df_windows, df_summary, summary_data = run_inference(MODEL_DIR, DATA_PATH)

    # Optional: Display summary
    if df_summary is not None:
        print("\nFile Summary:")
        print(df_summary.to_string())
        print("\n" + "=" * 60)
        print(summary_data[0].get("file") if summary_data else None)
