"""
================================================================================
 German Credit Pipeline with Provenance Tracking
================================================================================

Dataset : 1000 rows, label is binary (1=good credit, 2=bad credit).
Class distribution is IMBALANCED (700 / 300) — perfect for showcasing
SMOTE-based oversampling with full provenance.

Pipeline:
    Step 1 : Filter — keep moderate loans (credit_amount <= 7000)
    Step 2 : Filter — keep applicants under 60 years old
    Step 3 : Oversample — balance the label classes via SMOTE

Outputs:
  - processed_results/german/german_processed.csv
  - processed_results/german/provenance_report.txt
  - processed_results/german/full_provenance_tensor.npz
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.sparse import save_npz

from prov import Prov, ProvenanceTensor


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

INPUT_PATH  = "Datasets/german.csv"
OUTPUT_DIR  = "processed_results/german"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# 1. Load the raw dataset and prepare numeric features for SMOTE
# ----------------------------------------------------------------------

df_raw = pd.read_csv(INPUT_PATH)
print(f"Loaded {INPUT_PATH}: {df_raw.shape}")
print(f"Label distribution (raw): \n{df_raw['label'].value_counts().to_string()}")

# Map label 2 -> 0 (minority) so the binary target is 0/1.
# This is a value transformation, OUTSIDE our provenance pipeline.
# It doesn't change row identity, so it doesn't affect row-level provenance.
df_raw["label"] = df_raw["label"].map({1: 1, 2: 0})

# Keep only numeric columns for SMOTE (filters can use any column).
# Out-of-scope ops: drop categorical cols. Row identity is preserved.
numeric_cols = ["duration", "credit_amount", "inst_rate", "residence_time",
                "age", "num_credits", "dependants", "label"]
df0 = df_raw[numeric_cols].copy().reset_index(drop=True)
print(f"After numeric-column selection: {df0.shape}")


# ----------------------------------------------------------------------
# 2. Apply the pipeline
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("Applying pipeline with provenance tracking")
print("=" * 70)

provs = []
op_times = {}

# Step 1: filter credit_amount <= 7000
t = time.perf_counter()
df1, P1 = Prov.filter(df0, df0["credit_amount"] <= 7000)
op_times["filter_credit"] = time.perf_counter() - t
print(f"Step 1 (credit_amount<=7000): {len(df0)} -> {len(df1)} rows  "
      f"({op_times['filter_credit']*1000:.1f} ms)")
provs.append(P1)

# Step 2: filter age < 60
t = time.perf_counter()
df2, P2 = Prov.filter(df1, df1["age"] < 60)
op_times["filter_age"] = time.perf_counter() - t
print(f"Step 2 (age<60)             : {len(df1)} -> {len(df2)} rows  "
      f"({op_times['filter_age']*1000:.1f} ms)")
provs.append(P2)

# Step 3: SMOTE on label
print(f"\nClass distribution before SMOTE: "
      f"{df2['label'].value_counts().to_dict()}")

t = time.perf_counter()
df3, P3 = Prov.oversample(
    df2, target_col="label", k_neighbors=5, random_state=42,
)
op_times["oversample"] = time.perf_counter() - t
print(f"Step 3 (SMOTE oversample)   : {len(df2)} -> {len(df3)} rows  "
      f"({op_times['oversample']*1000:.1f} ms)")
print(f"Class distribution after SMOTE : "
      f"{df3['label'].value_counts().to_dict()}")
provs.append(P3)


# ----------------------------------------------------------------------
# 3. Compose
# ----------------------------------------------------------------------

t = time.perf_counter()
T_full = ProvenanceTensor.compose_chain(provs)
t_compose = time.perf_counter() - t
print(f"\nCompose chain of {len(provs)} tensors: {t_compose*1000:.1f} ms")
print(f"Full provenance: {T_full}")


# ----------------------------------------------------------------------
# 4. Query benchmarks
# ----------------------------------------------------------------------

rng = np.random.default_rng(0)
n_out = T_full.shape[0][0]
n_in  = T_full.shape[0][1]

sample_out = rng.choice(n_out, size=min(100, n_out), replace=False)
t = time.perf_counter()
for i in sample_out:
    _ = T_full.backward(int(i))
t_backward = time.perf_counter() - t

sample_in = rng.choice(n_in, size=100, replace=False)
t = time.perf_counter()
for j in sample_in:
    _ = T_full.forward(int(j))
t_forward = time.perf_counter() - t

print(f"\nQuery benchmarks (100 samples each):")
print(f"  Backward : {t_backward*1000/len(sample_out):.3f} ms/query")
print(f"  Forward  : {t_forward*1000/len(sample_in):.3f} ms/query")


# ----------------------------------------------------------------------
# 5. Storage analysis
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("Storage analysis")
print("=" * 70)

storage_summary = []
for name, P in zip(
    ["filter_credit", "filter_age", "oversample", "FULL CHAIN"],
    provs + [T_full],
):
    nnz = P.nnz
    shape = P.shape[0]
    dense_b  = shape[0] * shape[1]
    sparse_b = nnz * (4 + 4 + 1)
    ratio    = dense_b / max(sparse_b, 1)
    storage_summary.append((name, shape, nnz, dense_b, sparse_b, ratio))
    print(f"  {name:15s}  shape={shape}  nnz={nnz:6d}  "
          f"dense={dense_b/1e6:7.3f} MB  sparse={sparse_b/1e3:7.2f} KB  "
          f"ratio={ratio:8.1f}x")


# ----------------------------------------------------------------------
# 6. Trace demonstration
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("Traceability demonstration")
print("=" * 70)
orig_idx  = 0
synth_idx = len(df2) + 0   # first synthetic row

for label, i in [("original ", orig_idx), ("synthetic", synth_idx)]:
    if i >= n_out:
        continue
    sources = T_full.backward(i)[0]
    print(f"\nFinal row {i} ({label}) traces back to df0 positions: {list(sources)}")
    if len(sources) > 0:
        print(df0.iloc[sources][["age", "credit_amount", "duration", "label"]].to_string())


# ----------------------------------------------------------------------
# 7. Save outputs
# ----------------------------------------------------------------------

df3.reset_index(drop=True).to_csv(
    os.path.join(OUTPUT_DIR, "german_processed.csv"), index=False)
save_npz(os.path.join(OUTPUT_DIR, "full_provenance_tensor.npz"), T_full.tensor)

report_path = os.path.join(OUTPUT_DIR, "provenance_report.txt")
with open(report_path, "w") as f:
    f.write("German Credit Pipeline — Provenance Report\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Raw input          : {df_raw.shape[0]} rows × {df_raw.shape[1]} cols\n")
    f.write(f"After col select   : {df0.shape[0]} rows × {df0.shape[1]} cols\n")
    f.write(f"After filter credit: {len(df1)} rows\n")
    f.write(f"After filter age   : {len(df2)} rows\n")
    f.write(f"After oversample   : {len(df3)} rows\n\n")
    f.write("Timing per step (ms)\n")
    for k, v in op_times.items():
        f.write(f"  {k:15s} {v*1000:8.2f}\n")
    f.write(f"  {'compose_chain':15s} {t_compose*1000:8.2f}\n\n")
    f.write("Storage per tensor\n")
    for name, shape, nnz, dense_b, sparse_b, ratio in storage_summary:
        f.write(f"  {name:15s} shape={shape} nnz={nnz} "
                f"dense={dense_b/1e6:.3f}MB sparse={sparse_b/1e3:.2f}KB "
                f"ratio={ratio:.1f}x\n")
    f.write(f"\nBackward query : {t_backward*1000/len(sample_out):.3f} ms/query\n")
    f.write(f"Forward query  : {t_forward*1000/len(sample_in):.3f} ms/query\n")

print(f"\nSaved outputs to {OUTPUT_DIR}/")
