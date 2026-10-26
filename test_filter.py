"""
================================================================================
 Test suite for Prov.filter — Horizontal Reduction
================================================================================

Reference example: slide 6 of the mini-project.

Input  : 5 rows (Alice 24, Bob 17, Charlie 35, David 45, Eve 19).
Filter : Age >= 25   ->  keeps Charlie (pos 2) and David (pos 3).
Expected tensor of shape (2, 5):
        T[0, 2] = 1     (output row 0 <- input row 2 = Charlie)
        T[1, 3] = 1     (output row 1 <- input row 3 = David)

I test:
  1. Slide 6 example with boolean mask
  2. Same example with the string-query API (should yield identical tensor)
  3. Edge case: filter on a dataframe with non-trivial pandas index
  4. Sanity properties of the tensor (shape, sparsity, exactly one '1' per row)
"""

import numpy as np
import pandas as pd
from prov import Prov


# ======================================================================
# TEST 1 : Slide 6 example with boolean mask
# ======================================================================

print("=" * 70)
print("TEST 1 : Filter on slide 6 example (boolean mask)")
print("=" * 70)

df = pd.DataFrame({
    "Name":   ["Alice", "Bob", "Charlie", "David", "Eve"],
    "Age":    [24, 17, 35, 45, 19],
    "Salary": [70000, 40000, 120000, 110000, 50000],
})

print("\nInput:")
print(df)

output_df, prov = Prov.filter(df, df["Age"] >= 25)

print("\nOutput (Age >= 25):")
print(output_df)

# Inspect the ProvenanceTensor object
print(f"\nProvenance object: {prov}")
print(f"  n_inputs = {prov.n_inputs}      (filter is single-input)")
print(f"  shape    = {prov.shape}   (one shape per input dataset)")
print(f"  nnz      = {prov.nnz}            (one '1' per kept row)")

# Use the .tensor shortcut since filter has a single input
print("\nDense tensor (via .tensor shortcut):")
print(prov.tensor.toarray())

# Backward queries (output -> input)
print("\nBackward queries:")
for i in range(len(output_df)):
    # backward() returns a list with one array per input dataset.
    # For filter, the list has length 1.
    sources = prov.backward(i)
    print(f"  Output row {i} <- input rows {list(sources[0])}")

# Forward queries (input -> output)
print("\nForward queries:")
for j in range(len(df)):
    dst = prov.forward(j)  # dataset_index=0 by default
    name = df.iloc[j]["Name"]
    if len(dst) == 0:
        print(f"  Input row {j} ({name:7s}) -> filtered out")
    else:
        print(f"  Input row {j} ({name:7s}) -> output rows {list(dst)}")


# ======================================================================
# TEST 2 : Same filter via the string-query API
# ======================================================================

print("\n" + "=" * 70)
print("TEST 2 : Same filter via the string-query API")
print("=" * 70)

output_df2, prov2 = Prov.filter(df, "Age >= 25")

print("\nOutput:")
print(output_df2)

print(f"\nProvenance: {prov2}")

# Compare the two tensors element-wise.
# (prov.tensor != prov2.tensor) returns a sparse matrix of differences;
# if the two are identical, this matrix has zero non-zero entries.
identical = (prov.tensor != prov2.tensor).nnz == 0
print(f"\nTensors identical to TEST 1 ? {identical}")
assert identical, "Boolean mask and string query should yield identical tensors!"
print("  both APIs produce the same provenance tensor")


# ======================================================================
# TEST 3 : Filter on a dataframe with NON-TRIVIAL pandas index
# ======================================================================

print("\n" + "=" * 70)
print("TEST 3 : Filter on a re-indexed dataframe (important for chaining)")
print("=" * 70)
print("Use case: when chaining two filters, the input of the second filter")
print("has a non-contiguous pandas index from the first filter's output.")
print("The tensor must use POSITIONS (0..n-1), not pandas labels.")

# Simulate a df that already comes from a previous operation:
# its pandas index is [10, 20, 30, 40] instead of [0, 1, 2, 3].
df_reindexed = pd.DataFrame(
    {"x": [5, 15, 25, 35], "y": [100, 200, 300, 400]},
    index=[10, 20, 30, 40],
)
print("\nInput (with non-trivial index):")
print(df_reindexed)
print(f"  index labels: {list(df_reindexed.index)}")

out3, prov3 = Prov.filter(df_reindexed, df_reindexed["x"] >= 15)
print("\nOutput (x >= 15):")
print(out3)
print(f"  output pandas index: {list(out3.index)}  (labels preserved by pandas)")

print(f"\nDense tensor (uses POSITIONS, not labels):")
print(prov3.tensor.toarray())

# Verify: the tensor uses positions 1, 2, 3 (where x = 15, 25, 35)
# NOT pandas labels 20, 30, 40.
expected = np.array([
    [0, 1, 0, 0],   # output row 0 (x=15) <- input position 1
    [0, 0, 1, 0],   # output row 1 (x=25) <- input position 2
    [0, 0, 0, 1],   # output row 2 (x=35) <- input position 3
])
assert (prov3.tensor.toarray() == expected).all(), \
    "Tensor must use positions, not pandas labels"
print("  tensor correctly uses positional indices (robust to chaining)")


# ======================================================================
# TEST 4 : Structural properties of the filter tensor
# ======================================================================

print("\n" + "=" * 70)
print("TEST 4 : Structural properties of the filter tensor")
print("=" * 70)

# Build a larger filter for clearer sparsity figures
big_df = pd.DataFrame({"x": np.arange(100)})
out_big, prov_big = Prov.filter(big_df, big_df["x"] >= 80)

print(f"\nInput size  : {len(big_df)}")
print(f"Output size : {len(out_big)}  (kept x >= 80)")
print(f"Tensor shape: {prov_big.shape}")
print(f"Tensor nnz  : {prov_big.nnz}")

# Property 1: exactly one '1' per output row
T = prov_big.tensor
sums_per_row = np.asarray(T.sum(axis=1)).flatten()
print(f"\nProperty 1 - Sum per row (should all be 1):")
print(f"  min = {sums_per_row.min()}, max = {sums_per_row.max()}")
assert (sums_per_row == 1).all(), "Each output row should have exactly one source"
print("  each output row has exactly one input source")

# Property 2: each column has at most one '1' (since filter is injective)
sums_per_col = np.asarray(T.sum(axis=0)).flatten()
print(f"\nProperty 2 - Sum per column (should be 0 or 1):")
print(f"  unique values = {sorted(set(sums_per_col.tolist()))}")
assert set(sums_per_col.tolist()) <= {0, 1}, "Each input row maps to at most one output"
print("  each input row contributes to at most one output row")

# Property 3: storage efficiency
n_in, n_out = len(big_df), len(out_big)
dense_size = n_in * n_out                 # 1 byte per cell (bool-like)
sparse_size = prov_big.nnz * (4 + 4 + 1)  # int32 row + int32 col + int8 data per entry
ratio = dense_size / max(sparse_size, 1)
print(f"\nProperty 3 - Storage efficiency:")
print(f"  Dense storage  : {dense_size} bytes (filling factor = {prov_big.nnz}/{dense_size} = {prov_big.nnz/dense_size:.2%})")
print(f"  Sparse storage : {sparse_size} bytes")
print(f"  Compression ratio = {ratio:.1f}x")
