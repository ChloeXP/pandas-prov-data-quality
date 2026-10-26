"""
Tests for Prov.oversample on the slide 8 example.

Slide 8 input (5 rows):
  0  Age=25  Income=40000  Category=0
  1  Age=30  Income=50000  Category=0
  2  Age=35  Income=60000  Category=0
  3  Age=40  Income=70000  Category=1   <-- minority
  4  Age=45  Income=80000  Category=1   <-- minority

Class 0 has 3 samples, class 1 has 2 samples.
SMOTE will generate 1 synthetic sample (3 - 2 = 1) to balance.
"""

import pandas as pd
from prov import Prov


# ======================================================================
# 1. Slide 8 example
# ======================================================================

print("=" * 70)
print("TEST 1 : Oversample on slide 8 example")
print("=" * 70)

df = pd.DataFrame({
    "Age":      [25, 30, 35, 40, 45],
    "Income":   [40000, 50000, 60000, 70000, 80000],
    "Category": [0, 0, 0, 1, 1],
})

print("\nInput:")
print(df)
print(f"\nClass distribution:\n{df['Category'].value_counts()}")

# Run oversample with a fixed seed so the result is reproducible
output_df, prov = Prov.oversample(
    df, target_col="Category",
    k_neighbors=1,            # only 2 minority samples → max k=1
    random_state=42,
)

print("\nOutput:")
print(output_df)
print(f"\nClass distribution after oversampling:\n{output_df['Category'].value_counts()}")

print(f"\nProvenance: {prov}")
print(f"  shape    = {prov.shape}")
print(f"  n_inputs = {prov.n_inputs}")

# Show the dense tensor
print("\nDense tensor:")
print(prov.tensor.toarray())

# Backward queries
print("\nBackward queries (output -> sources):")
for i in range(len(output_df)):
    sources = prov.backward(i)[0]
    is_synth = i >= len(df)
    label = "synthetic" if is_synth else "original "
    print(f"  Output row {i} ({label}) <- input rows {list(sources)}")

# Forward queries
print("\nForward queries (input -> outputs):")
for j in range(len(df)):
    dst = prov.forward(j)
    print(f"  Input row {j} -> output rows {list(dst)}")


# ======================================================================
# 2. Slightly bigger example: more synthetics to verify the structure
# ======================================================================

print("\n" + "=" * 70)
print("TEST 2 : Oversample with k=3 on a more imbalanced dataset")
print("=" * 70)

df2 = pd.DataFrame({
    "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0,    10.0, 11.0, 12.0],
    "y": [1.0, 2.0, 1.5, 2.5, 1.0, 2.0,    10.0, 11.0, 10.5],
    "label": [0, 0, 0, 0, 0, 0,             1, 1, 1],
})
print(f"\nInput shape: {df2.shape}")
print(f"Class distribution:\n{df2['label'].value_counts()}")
print(f"Need to generate: {6 - 3} = 3 synthetic samples")

out2, prov2 = Prov.oversample(
    df2, target_col="label",
    k_neighbors=2,
    random_state=42,
)

print(f"\nOutput shape: {out2.shape}")
print(f"Class distribution after oversampling:\n{out2['label'].value_counts()}")

print(f"\nProvenance: {prov2}")

print("\nDense tensor:")
print(prov2.tensor.toarray())

print("\nBackward queries:")
for i in range(len(out2)):
    sources = prov2.backward(i)[0]
    is_synth = i >= len(df2)
    label = "synthetic" if is_synth else "original "
    src_features = [tuple(df2.iloc[s][['x', 'y']].values) for s in sources]
    print(f"  Output row {i} ({label}) <- input rows {list(sources)}  features: {src_features}")

# Quick sanity check: nnz must be n_input + 2 * n_synthetic
n_input = len(df2)
n_synth = len(out2) - n_input
expected_nnz = n_input + 2 * n_synth
print(f"\nSanity check: nnz = {prov2.nnz}  (expected {expected_nnz} = {n_input} + 2*{n_synth})")
assert prov2.nnz == expected_nnz, "nnz mismatch!"
print("  correct")

# Each synthetic row must have exactly 2 ones, each original exactly 1
T = prov2.tensor.toarray()
n_ones_per_row = T.sum(axis=1)
print(f"\nNumber of '1' per row: {list(n_ones_per_row)}")
print(f"  Originals (rows 0..{n_input-1})  : should all be 1")
print(f"  Synthetics (rows {n_input}..{len(out2)-1}) : should all be 2")
assert (n_ones_per_row[:n_input] == 1).all(), "some original has != 1 source"
assert (n_ones_per_row[n_input:] == 2).all(), "some synthetic has != 2 parents"
print("  correct")


# ======================================================================
# 3. Sanity: synthetic sample is on the segment between its two parents
# ======================================================================

print("\n" + "=" * 70)
print("TEST 3 : Synthetic features on parent segment (math verification)")
print("=" * 70)

import numpy as np

# Pick the first synthetic sample and check its features lie between its parents
synth_pos = n_input  # first synthetic
sources = prov2.backward(synth_pos)[0]
p1, p2 = sources[0], sources[1]

x_synth = out2.iloc[synth_pos][['x', 'y']].values
x_p1    = df2.iloc[p1][['x', 'y']].values
x_p2    = df2.iloc[p2][['x', 'y']].values

# x_synth = x_p1 + lambda * (x_p2 - x_p1) for some lambda in [0, 1]
# So (x_synth - x_p1) and (x_p2 - x_p1) must be parallel, and lambda in [0,1]
v1 = x_synth - x_p1
v2 = x_p2 - x_p1

# Compute lambda (assuming v2 != 0)
mask = v2 != 0
if mask.any():
    lams = v1[mask] / v2[mask]
    print(f"Synthetic row {synth_pos}: parents {p1}, {p2}")
    print(f"  Parent 1 features : {x_p1}")
    print(f"  Parent 2 features : {x_p2}")
    print(f"  Synthetic features: {x_synth}")
    print(f"  Recovered lambda (per axis): {lams}")
    assert np.allclose(lams, lams[0]), "synthetic not on parent segment"
    assert 0 <= lams[0] <= 1, "lambda out of [0, 1]"
    print(f"  Synthetic = parent1 + {lams[0]:.3f} * (parent2 - parent1)")
