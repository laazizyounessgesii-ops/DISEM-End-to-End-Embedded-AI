"""
DISEM – Intelligent Diagnosis of Electromechanical Systems
Authors: Youness LAAZIZ, Mamadou Bakary KEBE
Supervisor: Prof. Mohamed RAFIK
ENSET – 2025/2026

Academic project – proper citation required for any reuse.
GitHub: https://github.com/laazizyounessgesii-ops/DISEM-End-to-End-Embedded-AI
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # cache INFO/WARNING
import tensorflow as tf
from tensorflow import keras


# =========================
# USER PARAMETERS
# =========================
FS = 10000  # Hz

DATASET_DIR = r"C:\Users\Gros Info\Desktop\DISEM\dataset_csv_500x7"
LABELS_CSV = os.path.join(DATASET_DIR, "labels.csv")

OUT_DIR = os.path.join(DATASET_DIR, "_out_dnn3_fused")
SCALER_PATH = os.path.join(OUT_DIR, "scaler.pkl")
KERAS_MODEL_PATH = os.path.join(OUT_DIR, "dnn3_float32.keras")

# If you want to test INT8 TFLite too:
EVAL_TFLITE_INT8 = False
TFLITE_PATH = os.path.join(OUT_DIR, "dnn3_int8.tflite")

CLASS_NAMES = [f"C{i}" for i in range(1, 8)]  # C1..C7


# =========================
# FEATURE EXTRACTION (same as training)
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
# DATA LOADING
# =========================
def load_labels(labels_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    file_candidates = ["file", "filename", "file_name", "path", "filepath", "csv", "csv_file"]
    class_candidates = ["class_id", "class", "label", "y", "target"]

    file_col = next((c for c in file_candidates if c in df.columns), None)
    class_col = next((c for c in class_candidates if c in df.columns), None)

    if file_col is None or class_col is None:
        raise ValueError(f"labels.csv must contain file and class columns. Found: {df.columns.tolist()}")

    df = df.rename(columns={file_col: "file", class_col: "class_id"})
    return df[["file", "class_id"]]

def build_dataset(dataset_dir: str, labels_df: pd.DataFrame, fs: float):
    X_list, y_list = [], []

    for _, row in labels_df.iterrows():
        fname = str(row["file"])
        class_id = int(row["class_id"])
        y = class_id - 1  # 1..7 -> 0..6

        fpath = fname
        if not os.path.isabs(fpath):
            fpath = os.path.join(dataset_dir, fname)

        if not os.path.exists(fpath):
            fpath2 = os.path.join(dataset_dir, os.path.basename(fname))
            if os.path.exists(fpath2):
                fpath = fpath2
            else:
                raise FileNotFoundError(f"Missing CSV: {fpath}")

        df = pd.read_csv(fpath)
        df.columns = [str(c).strip().lower() for c in df.columns]

        im = df["im"].to_numpy()
        vib = df["vib"].to_numpy()

        X_list.append(extract_32_features(im, vib, fs))
        y_list.append(y)

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


# =========================
# PLOTTING HELPERS
# =========================
def plot_confusion_matrix(cm: np.ndarray, class_names, title: str, out_path: str, is_normalized: bool):
    fig = plt.figure(figsize=(8, 7))
    ax = plt.gca()
    ax.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(class_names)), class_names)

    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            text = f"{val:.2f}" if is_normalized else f"{int(val)}"
            ax.text(j, i, text, ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)

def plot_precision_recall_f1(report_dict: dict, class_names, out_path: str):
    precision, recall, f1 = [], [], []
    # keys will be "0","1",... when target_names=None
    for i in range(len(class_names)):
        key = str(i)
        precision.append(report_dict[key]["precision"])
        recall.append(report_dict[key]["recall"])
        f1.append(report_dict[key]["f1-score"])

    x = np.arange(len(class_names))
    width = 0.25

    fig = plt.figure(figsize=(10, 5))
    plt.bar(x - width, precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar(x + width, f1, width, label="F1-score")

    plt.xticks(x, class_names)
    plt.ylim(0.0, 1.05)
    plt.title("Per-class Precision / Recall / F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)

def plot_class_distribution(y_train, y_test, class_names, out_path: str):
    fig = plt.figure(figsize=(10, 4))
    x = np.arange(len(class_names))

    train_counts = np.bincount(y_train, minlength=len(class_names))
    test_counts = np.bincount(y_test, minlength=len(class_names))

    width = 0.35
    plt.bar(x - width/2, train_counts, width, label="Train")
    plt.bar(x + width/2, test_counts, width, label="Test")

    plt.xticks(x, class_names)
    plt.title("Class Distribution (Train vs Test)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)

def plot_overall_metrics(report_dict: dict, accuracy: float, out_path: str):
    # macro avg row exists in output_dict
    macro_p = report_dict["macro avg"]["precision"]
    macro_r = report_dict["macro avg"]["recall"]
    macro_f1 = report_dict["macro avg"]["f1-score"]

    names = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    vals = [accuracy, macro_p, macro_r, macro_f1]

    fig = plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(names)), vals)
    plt.ylim(0.0, 1.05)
    plt.xticks(np.arange(len(names)), names, rotation=15, ha="right")
    plt.title("Overall Metrics")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


# =========================
# OPTIONAL: TFLITE INT8 EVAL
# =========================
def tflite_predict_int8(tflite_path: str, X_scaled: np.ndarray):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    in_scale, in_zp = in_det["quantization"]

    preds = []
    for i in range(X_scaled.shape[0]):
        x = X_scaled[i:i+1].astype(np.float32)

        q = np.round(x / in_scale + in_zp).astype(np.int32)
        q = np.clip(q, -128, 127).astype(np.int8)

        interpreter.set_tensor(in_det["index"], q)
        interpreter.invoke()
        yq = interpreter.get_tensor(out_det["index"])  # int8
        preds.append(int(np.argmax(yq, axis=1)[0]))

    return np.array(preds, dtype=np.int64)


# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    labels_df = load_labels(LABELS_CSV)
    X, y = build_dataset(DATASET_DIR, labels_df, FS)

    # same split as training
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )

    scaler = joblib.load(SCALER_PATH)
    X_test_s = scaler.transform(X_test)

    if EVAL_TFLITE_INT8:
        y_pred = tflite_predict_int8(TFLITE_PATH, X_test_s)
        model_name = "INT8 TFLite"
    else:
        model = keras.models.load_model(KERAS_MODEL_PATH)
        probs = model.predict(X_test_s, verbose=0)
        y_pred = np.argmax(probs, axis=1)
        model_name = "Float32 Keras"

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    report_txt = classification_report(y_test, y_pred, digits=4, target_names=CLASS_NAMES)
    report_dict = classification_report(y_test, y_pred, output_dict=True)  # keys "0".."6"

    print(f"\n===== EVALUATION ({model_name}) =====")
    print("Test accuracy:", acc)
    print("\nConfusion matrix (counts):\n", cm)
    print("\nClassification report:\n", report_txt)

    # normalized CM
    cm_norm = cm.astype(np.float32)
    cm_norm = cm_norm / (cm_norm.sum(axis=1, keepdims=True) + 1e-12)

    # outputs
    p_cm_counts = os.path.join(OUT_DIR, "confusion_matrix_counts.png")
    p_cm_norm = os.path.join(OUT_DIR, "confusion_matrix_normalized.png")
    p_prf = os.path.join(OUT_DIR, "per_class_precision_recall_f1.png")
    p_dist = os.path.join(OUT_DIR, "class_distribution_train_test.png")
    p_overall = os.path.join(OUT_DIR, "overall_metrics.png")
    report_path = os.path.join(OUT_DIR, "classification_report.txt")

    plot_confusion_matrix(cm, CLASS_NAMES, f"Confusion Matrix (Counts) - {model_name}", p_cm_counts, is_normalized=False)
    plot_confusion_matrix(cm_norm, CLASS_NAMES, f"Confusion Matrix (Normalized) - {model_name}", p_cm_norm, is_normalized=True)
    plot_precision_recall_f1(report_dict, CLASS_NAMES, p_prf)
    plot_class_distribution(y_train, y_test, CLASS_NAMES, p_dist)
    plot_overall_metrics(report_dict, acc, p_overall)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {acc:.6f}\n\n")
        f.write("Confusion matrix (counts):\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification report:\n")
        f.write(report_txt)

    print("\nSaved report artifacts in:", OUT_DIR)
    print(" -", p_cm_counts)
    print(" -", p_cm_norm)
    print(" -", p_prf)
    print(" -", p_dist)
    print(" -", p_overall)
    print(" -", report_path)
    print("\nDONE.")


if __name__ == "__main__":
    main()
