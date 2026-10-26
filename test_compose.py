"""
================================================================================
 Task 2 — Pipeline composition tests
================================================================================

We test pipeline composition on cases of increasing complexity:

  1. filter -> filter                    (two single-input ops chained)
  2. filter -> filter -> filter          (compose_chain on N ops)
  3. filter -> oversample                (different single-input ops)
  4. merge -> filter                     (multi-input upstream)
  5. Verify mathematical equivalence:
     "compose then backward" must equal "backward through each step"

The algebraic identity we exercise throughout:
       T_compo = sign(T_down @ T_up)
"""

import numpy as np
import pandas as pd
from prov import Prov, ProvenanceTensor


# ======================================================================
# TEST 1 : filter -> filter (linear chain of two single-input ops)
# ======================================================================

print("=" * 70)
print("TEST 1 : filter -> filter")
print("=" * 70)

D0 = pd.DataFrame({"x": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
print(f"\nD0 (input) : {list(D0['x'])}")

# Step 1 : keep x >= 30
D1, prov1 = Prov.filter(D0, D0["x"] >= 30)
print(f"\nStep 1 (x>=30):")
print(f"  D1 = {list(D1['x'])}  (pandas index: {list(D1.index)})")
print(f"  T1 shape: {prov1.shape}, nnz={prov1.nnz}")

# Step 2 : keep x <= 70   --> we work on D1, so the mask must match D1's length
D2, prov2 = Prov.filter(D1, D1["x"] <= 70)
print(f"\nStep 2 (x<=70 on D1):")
print(f"  D2 = {list(D2['x'])}")
print(f"  T2 shape: {prov2.shape}, nnz={prov2.nnz}")

# Compose the two tensors
T_compo = ProvenanceTensor.compose(prov1, prov2)
print(f"\nComposed: {T_compo}")
print(f"  shape: {T_compo.shape}  (should be [(n_D2, n_D0)] = [({len(D2)}, {len(D0)})])")

print("\nComposed dense tensor (D2 rows × D0 columns):")
print(T_compo.tensor.toarray())

# Mathematical verification:
# T_compo[i, j] = 1  iff  row i of D2 comes from row j of D0
# For each output row of D2, the backward through D1 should agree.
print("\nVerification — backward via composed tensor vs via separate steps:")
for i in range(len(D2)):
    # Via composed tensor (direct D2 -> D0)
    direct = T_compo.backward(i)[0]
    # Via two-step chase (D2 -> D1 -> D0)
    mid = prov2.backward(i)[0]              # D2[i] -> D1[?]
    indirect = set()
    for m in mid:
        for src in prov1.backward(int(m))[0]:
            indirect.add(int(src))
    indirect = np.array(sorted(indirect))
    same = set(int(x) for x in direct) == set(int(x) for x in indirect)
    print(f"  D2[{i}] (x={D2.iloc[i]['x']}): direct={list(direct)} indirect={list(indirect)} match={same}")
    assert same, f"Backward mismatch on row {i}"
print("  All backward queries match")


# ======================================================================
# TEST 2 : filter -> filter -> filter (compose_chain on 3 ops)
# ======================================================================

print("\n" + "=" * 70)
print("TEST 2 : compose_chain on 3 filters")
print("=" * 70)

D0 = pd.DataFrame({"x": np.arange(20)})
print(f"\nD0: {list(D0['x'])}")

D1, P1 = Prov.filter(D0, D0["x"] >= 5)        # keep x in [5..19]
D2, P2 = Prov.filter(D1, D1["x"] < 15)        # keep x in [5..14]
D3, P3 = Prov.filter(D2, D2["x"] % 2 == 0)    # keep even x in [5..14] -> {6, 8, 10, 12, 14}

print(f"D1 (x>=5)  : {list(D1['x'])}")
print(f"D2 (x<15)  : {list(D2['x'])}")
print(f"D3 (x even): {list(D3['x'])}")

T_chain = ProvenanceTensor.compose_chain([P1, P2, P3])
print(f"\nChain-composed tensor: {T_chain}")

print("\nFor each row of D3, trace back to D0:")
for i in range(len(D3)):
    direct = T_chain.backward(i)[0]
    print(f"  D3[{i}] (x={D3.iloc[i]['x']}) <- D0 position {list(direct)}")

# Algebraic sanity: composing all three should give the same result as
# composing (P1, P2) then composing with P3.
T_step = ProvenanceTensor.compose(
    ProvenanceTensor.compose(P1, P2),
    P3,
)
identical = (T_chain.tensor != T_step.tensor).nnz == 0
print(f"\nLeft-fold associativity check: {identical}")
assert identical, "compose_chain should equal repeated compose"
print("  compose_chain is consistent with manual folding")


# ======================================================================
# TEST 3 : filter -> oversample
# ======================================================================

print("\n" + "=" * 70)
print("TEST 3 : filter -> oversample (two different single-input ops)")
print("=" * 70)

D0 = pd.DataFrame({
    "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0,  10.0, 11.0],
    "y": [1.0, 2.0, 1.5, 2.5, 1.0, 2.0,  10.0, 11.0],
    "lab": [0, 0, 0, 0, 0, 0,             1, 1],
})

# Step 1: drop x < 2 (removes the first sample only)
D1, P1 = Prov.filter(D0, D0["x"] >= 2.0)
print(f"\nAfter filter (x>=2): {len(D1)} rows")
print(D1)

# Step 2: SMOTE on the filtered dataset (minority is class 1 with 2 samples,
#         majority is class 0 with 5 samples -> need to generate 3 synth.)
D2, P2 = Prov.oversample(D1, target_col="lab", k_neighbors=1, random_state=42)
print(f"\nAfter oversample: {len(D2)} rows")
print(D2)

T_compo = ProvenanceTensor.compose(P1, P2)
print(f"\nComposed: {T_compo}")
print(f"\nDense composed tensor (D2 rows × D0 columns):")
print(T_compo.tensor.toarray())

print("\nBackward traces D2 -> D0:")
for i in range(len(D2)):
    direct = T_compo.backward(i)[0]
    is_synth = i >= len(D1)
    label = "synth" if is_synth else "orig "
    n_parents = len(direct)
    print(f"  D2[{i}] ({label}): {n_parents} parent(s) in D0 -> positions {list(direct)}")

# Sanity check: original rows should have exactly 1 parent in D0,
# synthetic rows should have exactly 2 parents in D0
T_dense = T_compo.tensor.toarray()
sums = T_dense.sum(axis=1)
n_orig_after_filter = len(D1)
print(f"\nNumber of D0-parents per row of D2:")
print(f"  Originals carried through (rows 0..{n_orig_after_filter-1}): {list(sums[:n_orig_after_filter])} (expected all 1)")
print(f"  Synthetic rows (rows {n_orig_after_filter}..{len(D2)-1}):     {list(sums[n_orig_after_filter:])} (expected all 2)")
assert (sums[:n_orig_after_filter] == 1).all()
assert (sums[n_orig_after_filter:] == 2).all()
print("  Parent counts preserved through composition")


# ======================================================================
# TEST 4 : merge -> filter (multi-input upstream, single-input downstream)
# ======================================================================

print("\n" + "=" * 70)
print("TEST 4 : merge -> filter (branch-wise composition)")
print("=" * 70)

A = pd.DataFrame({"key": [1, 2, 3, 4], "val_a": ["a", "b", "c", "d"]})
B = pd.DataFrame({"key": [2, 3, 4, 5], "val_b": [20, 30, 40, 50]})

print("\nA:")
print(A)
print("\nB:")
print(B)

# Step 1: inner join
J, Pmerge = Prov.merge(A, B, on="key", how="inner")
print("\nMerged J = A ⨝ B (on key):")
print(J)
print(f"Pmerge: {Pmerge}")

# Step 2: filter J by val_b >= 30
J2, Pfilter = Prov.filter(J, J["val_b"] >= 30)
print(f"\nFiltered J2 (val_b >= 30):")
print(J2)
print(f"Pfilter: {Pfilter}")

# Compose: should yield TWO tensors (one per merge branch)
T_compo = ProvenanceTensor.compose(Pmerge, Pfilter)
print(f"\nComposed: {T_compo}")
print(f"  shape: {T_compo.shape}")
print(f"  Expected: [(n_J2, len(A))=(2, 4), (n_J2, len(B))=(2, 4)]")

print("\nT_compo[0] (J2 rows × A columns):")
print(T_compo.tensors[0].toarray())
print("\nT_compo[1] (J2 rows × B columns):")
print(T_compo.tensors[1].toarray())

print("\nBackward traces J2 -> {A, B}:")
for i in range(len(J2)):
    sources = T_compo.backward(i)
    src_A = list(sources[0])
    src_B = list(sources[1])
    print(f"  J2[{i}] (key={J2.iloc[i]['key']}, val_b={J2.iloc[i]['val_b']}):  A={src_A}, B={src_B}")

# Sanity: each row of J2 should have exactly 1 parent in A and 1 in B
for i in range(len(J2)):
    sources = T_compo.backward(i)
    assert len(sources[0]) == 1 and len(sources[1]) == 1, \
        f"J2[{i}] should have one parent on each side"
print("  Each row of J2 traces back to exactly one (A, B) pair")


# ======================================================================
# TEST 5 : Stress test — random pipeline of 5 filters
# ======================================================================

print("\n" + "=" * 70)
print("TEST 5 : Random pipeline of 5 filters on 1000 rows")
print("=" * 70)

rng = np.random.default_rng(0)
D = pd.DataFrame({
    "a": rng.integers(0, 100, size=1000),
    "b": rng.integers(0, 100, size=1000),
    "c": rng.integers(0, 100, size=1000),
})

provs = []
current = D
masks = [
    current["a"] > 20,
    None,  # rebuilt after each step
]

# Pipeline: 5 random thresholds
thresholds = [(c, t) for c, t in zip(["a", "b", "c", "a", "b"], [20, 30, 40, 50, 25])]
for col, thr in thresholds:
    new, p = Prov.filter(current, current[col] > thr)
    provs.append(p)
    current = new

print(f"\nInput  : 1000 rows")
print(f"Output : {len(current)} rows (after 5 filters)")

T_full = ProvenanceTensor.compose_chain(provs)
print(f"Composed tensor: {T_full}")

# Verify: every output row has exactly one parent in D0 (chain of filters
# is injective)
sums = np.asarray(T_full.tensor.sum(axis=1)).flatten()
print(f"  Parents per row : min={sums.min()}, max={sums.max()}  (expected: all 1)")
assert (sums == 1).all()
print("  Chain of filters is injective (1 parent per output row)")

# Storage efficiency
T_dense_size = T_full.tensor.shape[0] * T_full.tensor.shape[1]    # bytes if dense bool
T_sparse_size = T_full.nnz * (4 + 4 + 1)                          # int32 + int32 + int8
ratio = T_dense_size / max(T_sparse_size, 1)
print(f"\nStorage on the composed tensor of shape {T_full.shape[0]}:")
print(f"  Dense  : {T_dense_size:,} bytes")
print(f"  Sparse : {T_sparse_size:,} bytes")
print(f"  Compression ratio : {ratio:.1f}x")
