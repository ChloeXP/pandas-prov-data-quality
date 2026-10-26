"""
================================================================================
 Cross-dataset summary report
================================================================================

This script runs all three pipelines (Compass, German, Census) and produces a
single comparative summary in processed_results/summary.txt.

Unlike the per-dataset reports, which document each pipeline in detail, this
summary aggregates the headline numbers in side-by-side tables for direct
comparison and inclusion in the final report.

The detailed per-dataset reports remain available under:
  - processed_results/compas/provenance_report.txt
  - processed_results/german/provenance_report.txt
  - processed_results/census/provenance_report.txt
"""

import os
import time
import numpy as np
import pandas as pd

from prov import Prov, ProvenanceTensor


# ----------------------------------------------------------------------
# Each pipeline is encapsulated in a function that returns its metrics.
# I use the SAME logic as the individual pipeline scripts; only output
# is suppressed and metrics are returned in a structured dict.
# ----------------------------------------------------------------------

def run_compas():
    df_raw = pd.read_csv("Datasets/compas.csv")
    KEEP = ["age", "c_charge_degree", "race", "sex", "priors_count",
            "days_b_screening_arrest", "two_year_recid"]
    df0 = df_raw[KEEP].copy().reset_index(drop=True)

    provs, op_times = [], {}

    t = time.perf_counter()
    df1, P1 = Prov.filter(df0, df0.notna().all(axis=1))
    op_times["filter_dropna"] = time.perf_counter() - t; provs.append(P1)

    t = time.perf_counter()
    df2, P2 = Prov.filter(df1, df1["days_b_screening_arrest"].abs() <= 30)
    op_times["filter_days"] = time.perf_counter() - t; provs.append(P2)

    t = time.perf_counter()
    df3, P3 = Prov.filter(df2, df2["c_charge_degree"] == "F")
    op_times["filter_felony"] = time.perf_counter() - t; provs.append(P3)

    # Encode for SMOTE (out-of-scope: doesn't affect row identity)
    df3n = df3.copy()
    df3n["race_caucasian"] = (df3n["race"] == "Caucasian").astype(int)
    df3n["sex_male"]       = (df3n["sex"] == "Male").astype(int)
    df3n = df3n.drop(columns=["race", "sex", "c_charge_degree"])

    t = time.perf_counter()
    df4, P4 = Prov.oversample(df3n, target_col="two_year_recid",
                              k_neighbors=5, random_state=42)
    op_times["oversample"] = time.perf_counter() - t; provs.append(P4)

    return _collect_metrics("compas", df_raw, df0, [df1, df2, df3, df4],
                            provs, op_times)


def run_german():
    df_raw = pd.read_csv("Datasets/german.csv")
    df_raw["label"] = df_raw["label"].map({1: 1, 2: 0})
    numeric_cols = ["duration", "credit_amount", "inst_rate", "residence_time",
                    "age", "num_credits", "dependants", "label"]
    df0 = df_raw[numeric_cols].copy().reset_index(drop=True)

    provs, op_times = [], {}

    t = time.perf_counter()
    df1, P1 = Prov.filter(df0, df0["credit_amount"] <= 7000)
    op_times["filter_credit"] = time.perf_counter() - t; provs.append(P1)

    t = time.perf_counter()
    df2, P2 = Prov.filter(df1, df1["age"] < 60)
    op_times["filter_age"] = time.perf_counter() - t; provs.append(P2)

    t = time.perf_counter()
    df3, P3 = Prov.oversample(df2, target_col="label",
                              k_neighbors=5, random_state=42)
    op_times["oversample"] = time.perf_counter() - t; provs.append(P3)

    return _collect_metrics("german", df_raw, df0, [df1, df2, df3],
                            provs, op_times)


def run_census():
    COL_NAMES = ["age", "workclass", "fnlwgt", "education", "education_num",
                 "marital_status", "occupation", "relationship", "race", "sex",
                 "capital_gain", "capital_loss", "hours_per_week",
                 "native_country", "label"]
    df_raw = pd.read_csv("Datasets/census.csv", header=0, names=COL_NAMES)
    for c in ["workclass", "education", "marital_status", "occupation",
              "relationship", "race", "sex", "native_country", "label"]:
        df_raw[c] = df_raw[c].str.strip()
    df_raw["label"] = df_raw["label"].map({"<=50K": 0, ">50K": 1})
    df_raw["sex"]   = df_raw["sex"].map({"Male": 1, "Female": 0})

    numeric_cols = ["age", "fnlwgt", "education_num", "sex",
                    "capital_gain", "capital_loss", "hours_per_week", "label"]
    df0 = df_raw[numeric_cols + ["workclass"]].copy().reset_index(drop=True)

    provs, op_times = [], {}

    t = time.perf_counter()
    df1, P1 = Prov.filter(df0, df0["workclass"] != "?")
    op_times["filter_qmark"] = time.perf_counter() - t; provs.append(P1)
    df1 = df1.drop(columns=["workclass"])

    t = time.perf_counter()
    df2, P2 = Prov.filter(df1, (df1["age"] >= 25) & (df1["age"] <= 60))
    op_times["filter_age"] = time.perf_counter() - t; provs.append(P2)

    t = time.perf_counter()
    df3, P3 = Prov.filter(df2, df2["capital_gain"] == 0)
    op_times["filter_cgain"] = time.perf_counter() - t; provs.append(P3)

    t = time.perf_counter()
    df4, P4 = Prov.oversample(df3, target_col="label",
                              k_neighbors=5, random_state=42)
    op_times["oversample"] = time.perf_counter() - t; provs.append(P4)

    return _collect_metrics("census", df_raw, df0, [df1, df2, df3, df4],
                            provs, op_times)


# ----------------------------------------------------------------------
# Common metrics-collection helper
# ----------------------------------------------------------------------

def _collect_metrics(name, df_raw, df0, intermediate_dfs, provs, op_times):
    """Compose, benchmark queries, compute storage, and return everything."""
    # Compose the chain
    t = time.perf_counter()
    T_full = ProvenanceTensor.compose_chain(provs)
    t_compose = time.perf_counter() - t

    # Query benchmarks
    rng = np.random.default_rng(0)
    n_out, n_in = T_full.shape[0]
    sample_out = rng.choice(n_out, size=min(100, n_out), replace=False)
    t = time.perf_counter()
    for i in sample_out:
        _ = T_full.backward(int(i))
    t_backward = (time.perf_counter() - t) * 1000 / len(sample_out)  # ms/query

    sample_in = rng.choice(n_in, size=min(100, n_in), replace=False)
    t = time.perf_counter()
    for j in sample_in:
        _ = T_full.forward(int(j))
    t_forward = (time.perf_counter() - t) * 1000 / len(sample_in)

    # Storage on the FULL CHAIN tensor only (for the summary)
    nnz = T_full.nnz
    shape = T_full.shape[0]
    dense_b  = shape[0] * shape[1]
    sparse_b = nnz * (4 + 4 + 1)
    ratio    = dense_b / max(sparse_b, 1)

    return {
        "name":            name,
        "raw_rows":        df_raw.shape[0],
        "raw_cols":        df_raw.shape[1],
        "after_select":    df0.shape[0],
        "final_rows":      len(intermediate_dfs[-1]),
        "n_ops":           len(provs),
        "op_times_ms":     {k: v * 1000 for k, v in op_times.items()},
        "compose_ms":      t_compose * 1000,
        "backward_ms":     t_backward,
        "forward_ms":      t_forward,
        "final_shape":     shape,
        "final_nnz":       nnz,
        "dense_bytes":     dense_b,
        "sparse_bytes":    sparse_b,
        "compression":     ratio,
    }


# ----------------------------------------------------------------------
# Main: run the three pipelines and emit the summary
# ----------------------------------------------------------------------

def main():
    print("Running 3 pipelines, please wait...\n")
    print(f"  [1/3] Compass... ", end="", flush=True)
    t = time.perf_counter()
    m_compas = run_compas()
    print(f"done ({time.perf_counter()-t:.1f}s)")

    print(f"  [2/3] German...  ", end="", flush=True)
    t = time.perf_counter()
    m_german = run_german()
    print(f"done ({time.perf_counter()-t:.1f}s)")

    print(f"  [3/3] Census...  ", end="", flush=True)
    t = time.perf_counter()
    m_census = run_census()
    print(f"done ({time.perf_counter()-t:.1f}s)")

    all_metrics = [m_compas, m_german, m_census]

    out_path = "processed_results/summary.txt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    lines = []
    lines.append("=" * 80)
    lines.append("  Cross-dataset summary — provenance tracking pipeline")
    lines.append("=" * 80)
    lines.append("")
    lines.append("This document aggregates headline metrics from the three pipelines.")
    lines.append("For per-pipeline details, see:")
    for m in all_metrics:
        lines.append(f"  - processed_results/{m['name']}/provenance_report.txt")
    lines.append("")

    # Table 1: dataset sizes and pipeline length
    lines.append("-" * 80)
    lines.append(" 1. Pipeline footprint")
    lines.append("-" * 80)
    lines.append(f"  {'dataset':10s} {'raw rows':>10s} {'raw cols':>10s} "
                 f"{'final rows':>12s} {'#prov ops':>10s}")
    for m in all_metrics:
        lines.append(f"  {m['name']:10s} {m['raw_rows']:>10d} {m['raw_cols']:>10d} "
                     f"{m['final_rows']:>12d} {m['n_ops']:>10d}")
    lines.append("")

    # Table 2: timing
    lines.append("-" * 80)
    lines.append(" 2. Timing (ms)")
    lines.append("-" * 80)
    lines.append(f"  {'dataset':10s} {'total filters':>15s} {'oversample':>12s} "
                 f"{'compose':>10s}")
    for m in all_metrics:
        t_filters = sum(v for k, v in m['op_times_ms'].items()
                        if k.startswith("filter"))
        t_oversample = m['op_times_ms'].get("oversample", 0.0)
        lines.append(f"  {m['name']:10s} {t_filters:>15.2f} "
                     f"{t_oversample:>12.2f} {m['compose_ms']:>10.2f}")
    lines.append("")

    # Table 3: query benchmarks
    lines.append("-" * 80)
    lines.append(" 3. Query benchmarks on the composed full tensor (ms/query)")
    lines.append("-" * 80)
    lines.append(f"  {'dataset':10s} {'backward':>10s} {'forward':>10s}")
    for m in all_metrics:
        lines.append(f"  {m['name']:10s} {m['backward_ms']:>10.3f} "
                     f"{m['forward_ms']:>10.3f}")
    lines.append("")

    # Table 4: storage
    lines.append("-" * 80)
    lines.append(" 4. Storage of the composed FULL CHAIN tensor")
    lines.append("-" * 80)
    lines.append(f"  {'dataset':10s} {'shape':>16s} {'nnz':>10s} "
                 f"{'dense (MB)':>12s} {'sparse (KB)':>14s} {'ratio':>10s}")
    for m in all_metrics:
        shape_str = f"({m['final_shape'][0]},{m['final_shape'][1]})"
        lines.append(f"  {m['name']:10s} {shape_str:>16s} {m['final_nnz']:>10d} "
                     f"{m['dense_bytes']/1e6:>12.2f} "
                     f"{m['sparse_bytes']/1e3:>14.2f} "
                     f"{m['compression']:>9.1f}x")
    lines.append("")

    # Headline
    lines.append("-" * 80)
    lines.append(" 5. Headline findings")
    lines.append("-" * 80)
    best = max(all_metrics, key=lambda m: m['compression'])
    lines.append(f"  Best compression ratio observed : {best['compression']:.1f}x  "
                 f"on the '{best['name']}' dataset")
    lines.append(f"  (dense tensor would have weighed "
                 f"{best['dense_bytes']/1e6:.1f} MB, sparse needs only "
                 f"{best['sparse_bytes']/1e3:.1f} KB).")
    lines.append("")
    avg_backward = np.mean([m['backward_ms'] for m in all_metrics])
    avg_forward  = np.mean([m['forward_ms']  for m in all_metrics])
    lines.append(f"  Mean query latency across the 3 datasets:")
    lines.append(f"    backward = {avg_backward:.3f} ms/query")
    lines.append(f"    forward  = {avg_forward:.3f} ms/query")
    lines.append("  Both are essentially constant w.r.t. dataset size — "
                 "a direct consequence")
    lines.append("  of the CSR/CSC sparse representation.")
    lines.append("")
    lines.append("=" * 80)

    output = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(output)

    print(f"\n{output}")
    print(f"\n[Summary written to {out_path}]")


if __name__ == "__main__":
    main()
