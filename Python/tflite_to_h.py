"""
DISEM – Intelligent Diagnosis of Electromechanical Systems
Authors: Youness LAAZIZ, Mamadou Bakary KEBE
Supervisor: Prof. Mohamed RAFIK
ENSET – 2025/2026

Academic project – proper citation required for any reuse.
GitHub: https://github.com/laazizyounessgesii-ops/DISEM-End-to-End-Embedded-AI
"""


# tflite_to_h.py
import os

TFLITE_PATH = r"C:\Users\Gros Info\Desktop\DISEM\dataset_csv_500x7\_out_dnn3_fused\dnn3_int8.tflite"
OUT_H       = r"C:\Users\Gros Info\Desktop\DISEM\Arduino\model_data.h"

var_name = "model_data"

with open(TFLITE_PATH, "rb") as f:
    data = f.read()

os.makedirs(os.path.dirname(OUT_H), exist_ok=True)

with open(OUT_H, "w", encoding="utf-8") as f:
    f.write("#pragma once\n\n")
    f.write(f"// Auto-generated from: {os.path.basename(TFLITE_PATH)}\n")
    f.write(f"// Size: {len(data)} bytes\n\n")
    f.write(f"alignas(16) const unsigned char {var_name}[] = {{\n")

    for i, b in enumerate(data):
        if i % 12 == 0:
            f.write("  ")
        f.write(f"0x{b:02x}, ")
        if i % 12 == 11:
            f.write("\n")

    if len(data) % 12 != 0:
        f.write("\n")

    f.write("};\n")
    f.write(f"const unsigned int {var_name}_len = {len(data)};\n")

print("OK ->", OUT_H)
