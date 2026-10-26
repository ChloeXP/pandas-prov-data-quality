"""
================================================================================
 Compass Pipeline with Provenance Tracking
================================================================================

This script applies a realistic data preparation pipeline on the COMPAS
recidivism dataset (Pro Publica) while tracking provenance at every step
using our Prov class. The four steps of the pipeline are:

    Step 1 : Filter — remove rows with NaN on key columns (equiv. dropna)
    Step 2 : Filter — keep only rows where |days_b_screening_arrest| <= 30
                      (standard fairness-ML convention on this dataset)
    Step 3 : Filter — keep only felony charges (c_charge_degree == 'F')
    Step 4 : Oversample — balance the two_year_recid classes via SMOTE

At each step we record the local provenance tensor, then compose the whole
chain to obtain a single tensor relating the final cleaned dataset to the
raw input.

Outputs:
  - processed_results/compas/compas_processed.csv
  - processed_results/compas/provenance_report.txt
"""

import os
import time
import numpy as np
import pandas as pd

from prov import Prov, ProvenanceTensor


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

INPUT_PATH  = "Datasets/compas.csv"
OUTPUT_DIR  = "processed_results/compas"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# 1. Load the raw dataset
# ----------------------------------------------------------------------

t0 = time.perf_counter()
df_raw = pd.read_csv(INPUT_PATH)
print(f"Loaded {INPUT_PATH}: {df_raw.shape[0]} rows × {df_raw.shape[1]} cols  "
      f"({time.perf_counter() - t0:.3f}s)")

# Keep only the columns we will use downstream — done WITHOUT provenance
# tracking because column selection is "vertical reduction", which is
# outside the four operations covered by our framework. This pre-step does
# not change row identity, so it does not affect the row-level provenance
# chain that follows.
KEEP_COLS = [
    "age", "c_charge_degree", "race", "sex", "priors_count",
    "days_b_screening_arrest", "two_year_recid",
]
df0 = df_raw[KEEP_COLS].copy().reset_index(drop=True)
print(f"After column selection: {df0.shape[0]} rows × {df0.shape[1]} cols")


# ----------------------------------------------------------------------
# 2. Apply the pipeline, tracking provenance at every step
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("Applying pipeline with provenance tracking")
print("=" * 70)

provs = []                              # provenance tensors, in order
op_times = {}                           # time spent on each Prov call

# Step 1: drop rows with any NaN
t = time.perf_counter()
mask1 = df0.notna().all(axis=1)
df1, P1 = Prov.filter(df0, mask1)
op_times["filter_dropna"] = time.perf_counter() - t
print(f"Step 1 (filter dropna)      : {len(df0)} -> {len(df1)} rows  "
      f"({op_times['filter_dropna']*1000:.1f} ms)")
provs.append(P1)

# Step 2: |days_b_screening_arrest| <= 30
t = time.perf_counter()
mask2 = df1["days_b_screening_arrest"].abs() <= 30
df2, P2 = Prov.filter(df1, mask2)
op_times["filter_days"] = time.perf_counter() - t
print(f"Step 2 (filter |days|<=30)  : {len(df1)} -> {len(df2)} rows  "
      f"({op_times['filter_days']*1000:.1f} ms)")
provs.append(P2)

# Step 3: keep only felony charges
t = time.perf_counter()
mask3 = df2["c_charge_degree"] == "F"
df3, P3 = Prov.filter(df2, mask3)
op_times["filter_felony"] = time.perf_counter() - t
print(f"Step 3 (filter c=='F')      : {len(df2)} -> {len(df3)} rows  "
      f"({op_times['filter_felony']*1000:.1f} ms)")
provs.append(P3)

# Step 4: SMOTE oversampling to balance two_year_recid
# SMOTE needs numeric features. We encode race, sex, c_charge_degree
# OUTSIDE the provenance pipeline because these are value transformations
# (out of our four supported operations) that do not change row identity.
df3_numeric = df3.copy()
df3_numeric["race_caucasian"] = (df3_numeric["race"] == "Caucasian").astype(int)
df3_numeric["sex_male"]       = (df3_numeric["sex"] == "Male").astype(int)
# We have already filtered on c_charge_degree, so it is constant — drop it
df3_numeric = df3_numeric.drop(columns=["race", "sex", "c_charge_degree"])

t = time.perf_counter()
df4, P4 = Prov.oversample(
    df3_numeric,
    target_col="two_year_recid",
    k_neighbors=5,
    random_state=42,
)
op_times["oversample"] = time.perf_counter() - t
print(f"Step 4 (oversample SMOTE)   : {len(df3_numeric)} -> {len(df4)} rows  "
      f"({op_times['oversample']*1000:.1f} ms)")
provs.append(P4)

# Verify the rebalanced class distribution
print(f"\nClass distribution after oversample:")
print(df4["two_year_recid"].value_counts().to_string())


# ----------------------------------------------------------------------
# 3. Compose the whole pipeline
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("Composing the pipeline (Task 2)")
print("=" * 70)

t = time.perf_counter()
T_full = ProvenanceTensor.compose_chain(provs)
t_compose = time.perf_counter() - t
print(f"Compose chain of {len(provs)} tensors: {t_compose*1000:.1f} ms")
print(f"\nFull provenance: {T_full}")
print(f"  Maps the {T_full.shape[0][0]} final rows back to the {T_full.shape[0][1]} input rows.")


# ----------------------------------------------------------------------
# 4. Provenance query benchmarks
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("Provenance query benchmarks")
print("=" * 70)

# Backward queries on 100 random output rows
rng = np.random.default_rng(0)
n_out = T_full.shape[0][0]
sample_out = rng.choice(n_out, size=min(100, n_out), replace=False)

t = time.perf_counter()
for i in sample_out:
    _ = T_full.backward(int(i))
t_backward = time.perf_counter() - t
print(f"Backward query (100 samples): mean {t_backward*1000/len(sample_out):.3f} ms/query")

# Forward queries on 100 random input rows
n_in = T_full.shape[0][1]
sample_in = rng.choice(n_in, size=100, replace=False)

t = time.perf_counter()
for j in sample_in:
    _ = T_full.forward(int(j))
t_forward = time.perf_counter() - t
print(f"Forward query  (100 samples): mean {t_forward*1000/len(sample_in):.3f} ms/query")


# ----------------------------------------------------------------------
# 5. Storage analysis
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("Storage analysis")
print("=" * 70)

storage_summary = []
for name, P in zip(
    ["filter_dropna", "filter_days", "filter_felony", "oversample", "FULL CHAIN"],
    provs + [T_full],
):
    nnz = P.nnz
    # All shapes from a single-input prov: list with 1 tuple
    shape = P.shape[0]
    dense_b  = shape[0] * shape[1]
    sparse_b = nnz * (4 + 4 + 1)        # int32 row + int32 col + int8 data
    ratio    = dense_b / max(sparse_b, 1)
    storage_summary.append((name, shape, nnz, dense_b, sparse_b, ratio))
    print(f"  {name:18s}  shape={shape}  nnz={nnz:6d}  "
          f"dense={dense_b/1e6:7.2f} MB  sparse={sparse_b/1e3:7.2f} KB  "
          f"ratio={ratio:8.1f}x")


# ----------------------------------------------------------------------
# 6. Demonstration: trace a row from final dataset back to raw
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("Demonstration: trace a specific row back to its raw source(s)")
print("=" * 70)

# Pick one original carried-through row and one synthetic row
orig_idx  = 0
synth_idx = len(df3_numeric) + 0   # first synthetic row

for label, i in [("original ", orig_idx), ("synthetic", synth_idx)]:
    sources = T_full.backward(i)[0]
    print(f"\nFinal row {i} ({label}) traces back to df0 positions: {list(sources)}")
    if len(sources) > 0:
        # df0 is what entered the chain (after column selection but before any Prov op)
        print(df0.iloc[sources][["age", "race", "sex", "priors_count", "two_year_recid"]])


# ----------------------------------------------------------------------
# 7. Save outputs
# ----------------------------------------------------------------------

# Reset index for a clean CSV
df4_out = df4.reset_index(drop=True)
out_csv = os.path.join(OUTPUT_DIR, "compas_processed.csv")
df4_out.to_csv(out_csv, index=False)
print(f"\nSaved processed dataset to {out_csv}")

# Write a textual provenance report
report_path = os.path.join(OUTPUT_DIR, "provenance_report.txt")
with open(report_path, "w") as f:
    f.write("Compass Pipeline — Provenance Report\n")
    f.write("=" * 70 + "\n\n")
    f.write("Pipeline\n")
    f.write("--------\n")
    f.write(f"  Raw input          : {df_raw.shape[0]} rows × {df_raw.shape[1]} cols\n")
    f.write(f"  After col select   : {df0.shape[0]} rows × {df0.shape[1]} cols  "
            f"(non-Prov op)\n")
    f.write(f"  After filter dropna: {len(df1)} rows\n")
    f.write(f"  After filter days  : {len(df2)} rows\n")
    f.write(f"  After filter felony: {len(df3)} rows\n")
    f.write(f"  After oversample   : {len(df4)} rows\n\n")
    f.write("Timing per step (ms)\n")
    f.write("--------------------\n")
    for k, v in op_times.items():
        f.write(f"  {k:18s} {v*1000:8.2f}\n")
    f.write(f"  {'compose_chain':18s} {t_compose*1000:8.2f}\n\n")
    f.write("Storage per tensor\n")
    f.write("------------------\n")
    f.write(f"  {'tensor':18s} {'shape':16s} {'nnz':>8s} {'dense (MB)':>12s} "
            f"{'sparse (KB)':>12s} {'ratio':>10s}\n")
    for name, shape, nnz, dense_b, sparse_b, ratio in storage_summary:
        f.write(f"  {name:18s} {str(shape):16s} {nnz:>8d} "
                f"{dense_b/1e6:>12.2f} {sparse_b/1e3:>12.2f} {ratio:>10.1f}x\n")
    f.write("\nQuery benchmarks (mean per query, 100 random samples)\n")
    f.write("-----------------------------------------------------\n")
    f.write(f"  Backward : {t_backward*1000/len(sample_out):.3f} ms\n")
    f.write(f"  Forward  : {t_forward*1000/len(sample_in):.3f} ms\n")

print(f"Saved provenance report to {report_path}")

# Save the composed sparse tensor (npz format)
from scipy.sparse import save_npz
tensor_path = os.path.join(OUTPUT_DIR, "full_provenance_tensor.npz")
save_npz(tensor_path, T_full.tensor)
print(f"Saved composed tensor to {tensor_path}")
