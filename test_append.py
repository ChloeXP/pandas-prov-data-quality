"""
Tests for the new architecture:
  1. Regression test : filter still works after refactoring
  2. New : append on the slide 10 example
"""

import pandas as pd
from prov import Prov, ProvenanceTensor


# ======================================================================
# 1. REGRESSION TEST : Filter on slide 6 example
# ======================================================================

print("=" * 70)
print("TEST 1 : Filter (slide 6) - regression after refactoring")
print("=" * 70)

df = pd.DataFrame({
    "Name":   ["Alice", "Bob", "Charlie", "David", "Eve"],
    "Age":    [24, 17, 35, 45, 19],
    "Salary": [70000, 40000, 120000, 110000, 50000],
})

print("\nInput:")
print(df)

output_df, prov = Prov.filter(df, df["Age"] >= 25)

print("\nOutput:")
print(output_df)

print(f"\nProvenance: {prov}")
print(f"  n_inputs = {prov.n_inputs}")
print(f"  shape    = {prov.shape}")

# .tensor shortcut still works for single-input ops
print("\nDense tensor (via .tensor shortcut):")
print(prov.tensor.toarray())

# Backward query : returns a list (one array per input dataset)
print("\nBackward queries:")
for i in range(len(output_df)):
    sources = prov.backward(i)
    print(f"  Output row {i} <- {[list(s) for s in sources]}")

# Forward query
print("\nForward queries:")
for j in range(len(df)):
    dst = prov.forward(j)  # dataset_index=0 by default
    name = df.iloc[j]["Name"]
    if len(dst) == 0:
        print(f"  Input row {j} ({name:7s}) -> filtered out")
    else:
        print(f"  Input row {j} ({name:7s}) -> output row {list(dst)}")


# ======================================================================
# 2. NEW : Append on slide 10 example
# ======================================================================

print("\n" + "=" * 70)
print("TEST 2 : Append (slide 10)")
print("=" * 70)

# D^l (4 rows)
Dl = pd.DataFrame({
    "ID":        [10, 20, 30, 40],
    "Birthdate": ["1996-07-12", "1994-03-08", None, "1987-11-23"],
    "Gender":    ["F", "M", "F", "M"],
    "Postcode":  [90210, None, 12345, 67890],
})

# D^r (2 rows)
Dr = pd.DataFrame({
    "ID":   [20, 40],
    "Name": ["Alice", "Bob"],
})

print("\nInput Dl:")
print(Dl)
print("\nInput Dr:")
print(Dr)

# Run append with provenance tracking
output_df, prov = Prov.append([Dl, Dr], ignore_index=True)

print("\nOutput (Dl ⊎ Dr):")
print(output_df)

print(f"\nProvenance: {prov}")
print(f"  n_inputs = {prov.n_inputs}")
print(f"  shapes   = {prov.shape}")
print(f"  total nnz = {prov.nnz}")

# Show both tensors as dense
print("\nDense tensors:")
for k, T in enumerate(prov.to_dense()):
    print(f"\n  T_{k} (output × input_{k}):")
    print(T)

# Backward queries: for each output row, list sources from each input dataset
print("\nBackward queries (sortie -> [sources Dl, sources Dr]):")
for i in range(len(output_df)):
    sources = prov.backward(i)
    src_str = " | ".join(
        f"D{['l','r'][k]}={list(s)}" if len(s) > 0 else f"D{['l','r'][k]}=[]"
        for k, s in enumerate(sources)
    )
    print(f"  Output row {i}: {src_str}")

# Forward queries on Dl
print("\nForward queries on Dl:")
for j in range(len(Dl)):
    dst = prov.forward(j, dataset_index=0)
    print(f"  Dl row {j} -> output row {list(dst)}")

# Forward queries on Dr
print("\nForward queries on Dr:")
for j in range(len(Dr)):
    dst = prov.forward(j, dataset_index=1)
    print(f"  Dr row {j} -> output row {list(dst)}")


# ======================================================================
# 3. EDGE CASE : Append of 3 datasets
# ======================================================================

print("\n" + "=" * 70)
print("TEST 3 : Append of 3 datasets (sanity check)")
print("=" * 70)

A = pd.DataFrame({"x": [1, 2]})
B = pd.DataFrame({"x": [3]})
C = pd.DataFrame({"x": [4, 5, 6]})

out, prov = Prov.append([A, B, C], ignore_index=True)
print(f"\nOutput length = {len(out)}  (expected: 2+1+3 = 6)")
print(out)
print(f"\nProvenance shapes: {prov.shape}")
print(f"  Expected: [(6, 2), (6, 1), (6, 3)]")

# Backward on row 3 (should come from C, position 0)
print(f"\nbackward(3) = {[list(s) for s in prov.backward(3)]}")
print(f"  Expected: [[], [], [0]]  (row 3 of output = row 0 of C)")
