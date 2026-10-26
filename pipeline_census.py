"""
================================================================================
 Census (UCI Adult) Pipeline with Provenance Tracking
================================================================================

Dataset : 32561 rows, binary label (<=50K or >50K).
Class distribution is significantly imbalanced (~75% vs 25%).

Pipeline:
    Step 1 : Filter — drop rows containing '?' (missing-value sentinel)
    Step 2 : Filter — keep working-age adults (age >= 25 AND age <= 60)
    Step 3 : Filter — keep applicants with capital-gain == 0
                      (focus on the bulk of the population)
    Step 4 : Oversample — balance the >50K vs <=50K classes via SMOTE

This is our largest dataset and showcases the scalability of sparse
provenance tracking.

Outputs:
  - processed_results/census/census_processed.csv
  - processed_results/census/provenance_report.txt
  - processed_results/census/full_provenance_tensor.npz
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

INPUT_PATH = "Datasets/census.csv"
OUTPUT_DIR = "processed_results/census"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# 1. Load and prepare the raw dataset
# ----------------------------------------------------------------------

# Census CSV has numeric column names 0..14; assign meaningful labels.
COL_NAMES = ["age", "workclass", "fnlwgt", "education", "education_num",
             "marital_status", "occupation", "relationship", "race", "sex",
             "capital_gain", "capital_loss", "hours_per_week",
             "native_country", "label"]

df_raw = pd.read_csv(INPUT_PATH, header=0, names=COL_NAMES)
print(f"Loaded {INPUT_PATH}: {df_raw.shape}")

# Trim whitespace on string columns and recode label to binary 0/1.
# These are value transformations — out-of-scope for our provenance
# pipeline, but they don't change row identity.
for c in ["workclass", "education", "marital_status", "occupation",
          "relationship", "race", "sex", "native_country", "label"]:
    df_raw[c] = df_raw[c].str.strip()

df_raw["label"] = df_raw["label"].map({"<=50K": 0, ">50K": 1})
df_raw["sex"]   = df_raw["sex"].map({"Male": 1, "Female": 0})
print(f"Label distribution (raw): {df_raw['label'].value_counts().to_dict()}")

# Keep only numeric columns for SMOTE compatibility downstream.
numeric_cols = ["age", "fnlwgt", "education_num", "sex",
                "capital_gain", "capital_loss", "hours_per_week", "label"]
# We keep workclass briefly because we need to filter on '?' first.
df0 = df_raw[numeric_cols + ["workclass"]].copy().reset_index(drop=True)
print(f"After column selection: {df0.shape}")


# ----------------------------------------------------------------------
# 2. Apply the pipeline
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("Applying pipeline with provenance tracking")
print("=" * 70)

provs = []
op_times = {}

# Step 1: drop rows where workclass == '?'
t = time.perf_counter()
df1, P1 = Prov.filter(df0, df0["workclass"] != "?")
op_times["filter_qmark"] = time.perf_counter() - t
print(f"Step 1 (workclass != '?')   : {len(df0)} -> {len(df1)} rows  "
      f"({op_times['filter_qmark']*1000:.1f} ms)")
provs.append(P1)

# After this filter, we can safely drop workclass (out-of-scope column drop).
df1 = df1.drop(columns=["workclass"])

# Step 2: working-age adults
t = time.perf_counter()
df2, P2 = Prov.filter(df1, (df1["age"] >= 25) & (df1["age"] <= 60))
op_times["filter_age"] = time.perf_counter() - t
print(f"Step 2 (25 <= age <= 60)    : {len(df1)} -> {len(df2)} rows  "
      f"({op_times['filter_age']*1000:.1f} ms)")
provs.append(P2)

# Step 3: capital_gain == 0
t = time.perf_counter()
df3, P3 = Prov.filter(df2, df2["capital_gain"] == 0)
op_times["filter_cgain"] = time.perf_counter() - t
print(f"Step 3 (capital_gain == 0)  : {len(df2)} -> {len(df3)} rows  "
      f"({op_times['filter_cgain']*1000:.1f} ms)")
provs.append(P3)

# Step 4: SMOTE on the label
print(f"\nClass distribution before SMOTE: "
      f"{df3['label'].value_counts().to_dict()}")

t = time.perf_counter()
df4, P4 = Prov.oversample(
    df3, target_col="label", k_neighbors=5, random_state=42,
)
op_times["oversample"] = time.perf_counter() - t
print(f"Step 4 (SMOTE oversample)   : {len(df3)} -> {len(df4)} rows  "
      f"({op_times['oversample']*1000:.1f} ms)")
print(f"Class distribution after SMOTE : "
      f"{df4['label'].value_counts().to_dict()}")
provs.append(P4)


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

sample_out = rng.choice(n_out, size=200, replace=False)
t = time.perf_counter()
for i in sample_out:
    _ = T_full.backward(int(i))
t_backward = time.perf_counter() - t

sample_in = rng.choice(n_in, size=200, replace=False)
t = time.perf_counter()
for j in sample_in:
    _ = T_full.forward(int(j))
t_forward = time.perf_counter() - t

print(f"\nQuery benchmarks (200 samples each):")
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
    ["filter_qmark", "filter_age", "filter_cgain", "oversample", "FULL CHAIN"],
    provs + [T_full],
):
    nnz = P.nnz
    shape = P.shape[0]
    dense_b  = shape[0] * shape[1]
    sparse_b = nnz * (4 + 4 + 1)
    ratio    = dense_b / max(sparse_b, 1)
    storage_summary.append((name, shape, nnz, dense_b, sparse_b, ratio))
    print(f"  {name:15s}  shape={shape}  nnz={nnz:7d}  "
          f"dense={dense_b/1e6:8.2f} MB  sparse={sparse_b/1e3:8.2f} KB  "
          f"ratio={ratio:8.1f}x")


# ----------------------------------------------------------------------
# 6. Trace demonstration
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("Traceability demonstration")
print("=" * 70)
orig_idx  = 0
synth_idx = len(df3) + 0   # first synthetic row

for label, i in [("original ", orig_idx), ("synthetic", synth_idx)]:
    if i >= n_out:
        continue
    sources = T_full.backward(i)[0]
    print(f"\nFinal row {i} ({label}) traces back to df0 positions: {list(sources)}")
    if len(sources) > 0:
        print(df0.iloc[sources][["age", "education_num", "hours_per_week", "label"]].to_string())


# ----------------------------------------------------------------------
# 7. Save outputs
# ----------------------------------------------------------------------

df4.reset_index(drop=True).to_csv(
    os.path.join(OUTPUT_DIR, "census_processed.csv"), index=False)
save_npz(os.path.join(OUTPUT_DIR, "full_provenance_tensor.npz"), T_full.tensor)

report_path = os.path.join(OUTPUT_DIR, "provenance_report.txt")
with open(report_path, "w") as f:
    f.write("Census (Adult) Pipeline — Provenance Report\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Raw input          : {df_raw.shape[0]} rows × {df_raw.shape[1]} cols\n")
    f.write(f"After col select   : {df0.shape[0]} rows × {df0.shape[1]} cols\n")
    f.write(f"After filter '?'   : {len(df1)} rows\n")
    f.write(f"After filter age   : {len(df2)} rows\n")
    f.write(f"After filter cgain : {len(df3)} rows\n")
    f.write(f"After oversample   : {len(df4)} rows\n\n")
    f.write("Timing per step (ms)\n")
    for k, v in op_times.items():
        f.write(f"  {k:15s} {v*1000:8.2f}\n")
    f.write(f"  {'compose_chain':15s} {t_compose*1000:8.2f}\n\n")
    f.write("Storage per tensor\n")
    for name, shape, nnz, dense_b, sparse_b, ratio in storage_summary:
        f.write(f"  {name:15s} shape={shape} nnz={nnz} "
                f"dense={dense_b/1e6:.2f}MB sparse={sparse_b/1e3:.2f}KB "
                f"ratio={ratio:.1f}x\n")
    f.write(f"\nBackward query : {t_backward*1000/len(sample_out):.3f} ms/query\n")
    f.write(f"Forward query  : {t_forward*1000/len(sample_in):.3f} ms/query\n")

print(f"\nSaved outputs to {OUTPUT_DIR}/")
