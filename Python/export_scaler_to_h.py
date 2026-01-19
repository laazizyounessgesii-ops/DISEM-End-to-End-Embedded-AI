"""
DISEM – Intelligent Diagnosis of Electromechanical Systems
Authors: Youness LAAZIZ, Mamadou Bakary KEBE
Supervisor: Prof. Mohamed RAFIK
ENSET – 2025/2026

Academic project – proper citation required for any reuse.
GitHub: https://github.com/laazizyounessgesii-ops/DISEM-End-to-End-Embedded-AI
"""


# export_scaler_to_h.py
import os
import numpy as np
import joblib

SCALER_PKL = r"C:\Users\Gros Info\Desktop\DISEM\dataset_csv_500x7\_out_dnn3_fused\scaler.pkl"
OUT_H      = r"C:\Users\Gros Info\Desktop\DISEM\Arduino\scaler_params.h"

scaler = joblib.load(SCALER_PKL)

mean_  = np.asarray(scaler.mean_,  dtype=np.float32)
scale_ = np.asarray(scaler.scale_, dtype=np.float32)

def fmt(v: float) -> str:
    return f"{float(v):.9e}f"   # évite le bug "26702140f"

os.makedirs(os.path.dirname(OUT_H), exist_ok=True)

with open(OUT_H, "w", encoding="utf-8") as f:
    f.write("#pragma once\n\n")
    f.write(f"#define SCALER_DIM {len(mean_)}\n\n")

    f.write("static const float scaler_mean[SCALER_DIM] = {\n  ")
    f.write(", ".join(fmt(x) for x in mean_))
    f.write("\n};\n\n")

    f.write("static const float scaler_scale[SCALER_DIM] = {\n  ")
    f.write(", ".join(fmt(x) for x in scale_))
    f.write("\n};\n")

print("OK ->", OUT_H)
