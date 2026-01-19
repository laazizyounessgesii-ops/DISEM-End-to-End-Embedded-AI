"""
DISEM – Intelligent Diagnosis of Electromechanical Systems
Authors: Youness LAAZIZ, Mamadou Bakary KEBE
Supervisor: Prof. Mohamed RAFIK
ENSET – 2025/2026

Academic project – proper citation required for any reuse.
GitHub: https://github.com/laazizyounessgesii-ops/DISEM-End-to-End-Embedded-AI
"""


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import joblib

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow import keras
from tensorflow.keras import layers


# =========================
# PARAMETERS
# =========================
FS = 10000  # Hz

DATASET_DIR = r"C:\Users\Gros Info\Desktop\DISEM\dataset_csv_500x7"
LABELS_CSV = os.path.join(DATASET_DIR, "labels.csv")
OUT_DIR = os.path.join(DATASET_DIR, "_out_dnn3_fused")

os.makedirs(OUT_DIR, exist_ok=True)

SAVE_DEBUG_FEATURES = True
NUM_CLASSES = 7  # C1..C7


# =========================
# FEATURE EXTRACTION
# =========================
EPS = 1e-12

def _safe_std(x: np.ndarray) -> float:
    s = float(np.std(x))
    return s if s > EPS else EPS

def _skewness(x: np.ndarray) -> float:
    mu = float(np.mean(x))
    s = _safe_std(x)
    z = (x - mu) / s
    return float(np.mean(z ** 3))

def _kurtosis_excess(x: np.ndarray) -> float:
    mu = float(np.mean(x))
    s = _safe_std(x)
    z = (x - mu) / s
    return float(np.mean(z ** 4) - 3.0)

def _mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))

def _bandpower(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))

def _spectral_features(x: np.ndarray, fs: float):
    x = x.astype(np.float64)
    x = x - np.mean(x)

    n = len(x)
    X = np.fft.rfft(x)
    mag2 = (np.abs(X) ** 2)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    mag2_no_dc = mag2.copy()
    if len(mag2_no_dc) > 0:
        mag2_no_dc[0] = 0.0
    dom_idx = int(np.argmax(mag2_no_dc)) if len(mag2_no_dc) > 1 else 0
    dom_freq = float(freqs[dom_idx]) if len(freqs) > 0 else 0.0

    denom = float(np.sum(mag2)) + EPS
    centroid = float(np.sum(freqs * mag2) / denom)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag2) / denom))

    psd_like = mag2 / (n + EPS)

    nyq = fs / 2.0
    b1_hi = min(200.0, nyq)
    b2_hi = min(1000.0, nyq)
    b3_hi = min(4000.0, nyq)

    bp1 = _bandpower(freqs, psd_like, 0.0, b1_hi)
    bp2 = _bandpower(freqs, psd_like, 200.0, b2_hi) if b2_hi > 200.0 else 0.0
    bp3 = _bandpower(freqs, psd_like, 1000.0, b3_hi) if b3_hi > 1000.0 else 0.0

    return dom_freq, centroid, bandwidth, bp1, bp2, bp3

def extract_16_features_one_channel(x: np.ndarray, fs: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)

    mean = float(np.mean(x))
    std = float(np.std(x))
    rms = _rms(x)
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    ptp = float(xmax - xmin)
    med = float(np.median(x))
    mad = _mad(x)
    skew = _skewness(x)
    kurt = _kurtosis_excess(x)

    domf, cent, bw, bp1, bp2, bp3 = _spectral_features(x, fs)

    return np.array(
        [mean, std, rms, xmin, xmax, ptp, med, mad, skew, kurt,
         domf, cent, bw, bp1, bp2, bp3],
        dtype=np.float32
    )

def extract_32_features(im: np.ndarray, vib: np.ndarray, fs: float) -> np.ndarray:
    return np.concatenate(
        [extract_16_features_one_channel(im, fs),
         extract_16_features_one_channel(vib, fs)],
        axis=0
    ).astype(np.float32)


# =========================
# DATA LOADING (NO labels.csv needed)
# =========================
def build_dataset(dataset_dir: str, fs: float):
    csv_files = sorted(glob.glob(os.path.join(DATASET_DIR, "C*_*.csv")))
    print("Found CSV files:", len(csv_files))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in: {dataset_dir}")

    print("Found CSV files:", len(csv_files))

    X_list, y_list, file_list = [], [], []

    for fpath in csv_files:
        df = pd.read_csv(fpath)
        df.columns = [str(c).strip().lower() for c in df.columns]

        needed = {"im", "vib", "class_id"}
        if not needed.issubset(set(df.columns)):
            raise ValueError(f"{os.path.basename(fpath)} missing columns. Found={df.columns.tolist()}")

        class_id = int(df["class_id"].iloc[0])  # 1..7
        y = class_id - 1  # 0..6
        if y < 0 or y >= NUM_CLASSES:
            raise ValueError(f"Invalid class_id={class_id} in {os.path.basename(fpath)}")

        im = df["im"].to_numpy()
        vib = df["vib"].to_numpy()

        feats = extract_32_features(im, vib, fs)

        X_list.append(feats)
        y_list.append(y)
        file_list.append(os.path.basename(fpath))

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, file_list


# =========================
# MODEL
# =========================
def make_model(input_dim: int, num_classes: int = NUM_CLASSES) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,), name="features"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# =========================
# INT8 TFLITE CONVERSION
# =========================
def convert_to_int8_tflite(model: keras.Model, X_rep: np.ndarray, out_path: str):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep_data_gen():
        for i in range(min(len(X_rep), 200)):
            yield [X_rep[i:i + 1].astype(np.float32)]

    converter.representative_dataset = rep_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)

def save_quant_params(tflite_path: str, out_json: str):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    info = {
        "input": {
            "name": in_detail["name"],
            "dtype": str(in_detail["dtype"]),
            "shape": [int(x) for x in in_detail["shape"]],
            "quantization": {"scale": float(in_detail["quantization"][0]), "zero_point": int(in_detail["quantization"][1])},
        },
        "output": {
            "name": out_detail["name"],
            "dtype": str(out_detail["dtype"]),
            "shape": [int(x) for x in out_detail["shape"]],
            "quantization": {"scale": float(out_detail["quantization"][0]), "zero_point": int(out_detail["quantization"][1])},
        },
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


# =========================
# MAIN
# =========================
def main():
    print("Dataset:", DATASET_DIR)
    X, y, file_list = build_dataset(DATASET_DIR, FS)
    print("X shape:", X.shape, "y shape:", y.shape)

    X_train, X_tmp, y_train, y_tmp, files_train, files_tmp = train_test_split(
        X, y, file_list, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test, files_val, files_test = train_test_split(
        X_tmp, y_tmp, files_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )

    print("Split sizes:", "train", len(y_train), "val", len(y_val), "test", len(y_test))

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    scaler_path = os.path.join(OUT_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print("Saved:", scaler_path)

    if SAVE_DEBUG_FEATURES:
        dbg_path = os.path.join(OUT_DIR, "features_train_sample.csv")
        dbg_df = pd.DataFrame(X_train_s[:50], columns=[f"f{i:02d}" for i in range(X_train_s.shape[1])])
        dbg_df.insert(0, "class", y_train[:50])
        dbg_df.insert(0, "file", np.array(files_train[:50], dtype=object))
        dbg_df.to_csv(dbg_path, index=False)
        print("Saved:", dbg_path)

    model = make_model(input_dim=X_train_s.shape[1], num_classes=NUM_CLASSES)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True)
    ]

    model.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=300,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    y_pred = np.argmax(model.predict(X_test_s, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n===== TEST RESULTS =====")
    print("Test accuracy:", acc)
    print("\nConfusion matrix:\n", cm)
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))

    keras_path = os.path.join(OUT_DIR, "dnn3_float32.keras")
    model.save(keras_path)
    print("Saved:", keras_path)

    tflite_path = os.path.join(OUT_DIR, "dnn3_int8.tflite")
    convert_to_int8_tflite(model, X_train_s, tflite_path)
    print("Saved:", tflite_path)

    qinfo_path = os.path.join(OUT_DIR, "dnn3_int8_quant_params.json")
    save_quant_params(tflite_path, qinfo_path)
    print("Saved:", qinfo_path)

    print("\nDONE. Output folder:", OUT_DIR)

if __name__ == "__main__":
    main()
