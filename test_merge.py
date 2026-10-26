"""
Tests for Prov.merge.

I test:
  1. Inner join on the slide 9 example (the textbook case)
  2. Left outer join (to validate NaN handling in tracker columns)
  3. 1-to-N join (one left row matching several right rows)
  4. Defensive check: tracker column collision raises ValueError
"""

import numpy as np
import pandas as pd
from prov import Prov


# ======================================================================
# TEST 1 : Inner join on slide 9 example
# ======================================================================

print("=" * 70)
print("TEST 1 : Inner join (slide 9)")
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

output_df, prov = Prov.merge(Dl, Dr, on="ID", how="inner")

print("\nOutput (Dl ⨝ Dr on ID, inner):")
print(output_df)

print(f"\nProvenance: {prov}")

print("\nDense tensors:")
for k, T in enumerate(prov.to_dense()):
    side = "left (Dl)" if k == 0 else "right (Dr)"
    print(f"\n  T_{k} (output × {side}):")
    print(T)

print("\nExpected (from slide 12):")
print("  T_left  : 1 at (0,1), 1 at (1,3)  -> rows of Dl with ID=20 and ID=40")
print("  T_right : 1 at (0,0), 1 at (1,1)  -> rows of Dr (Alice and Bob)")

# Backward queries: both sides should have a parent for each output row
print("\nBackward queries (output -> [sources Dl, sources Dr]):")
for i in range(len(output_df)):
    sources = prov.backward(i)
    print(f"  Output row {i}: Dl={list(sources[0])}, Dr={list(sources[1])}")

# Sanity check: every output row has exactly 2 parents (1 in each side)
T_l_dense = prov.to_dense()[0]
T_r_dense = prov.to_dense()[1]
sums_l = T_l_dense.sum(axis=1)
sums_r = T_r_dense.sum(axis=1)
print(f"\nSanity check (inner join): each output row should have exactly 1 parent in each side")
print(f"  Sum per row in T_left  = {list(sums_l)}  (expected: all 1)")
print(f"  Sum per row in T_right = {list(sums_r)}  (expected: all 1)")
assert (sums_l == 1).all() and (sums_r == 1).all()
print("  correct")


# ======================================================================
# TEST 2 : Left outer join
# ======================================================================

print("\n" + "=" * 70)
print("TEST 2 : Left outer join")
print("=" * 70)
print("All 4 rows of Dl should be kept; only 2 will have a match in Dr.")

output_df2, prov2 = Prov.merge(Dl, Dr, on="ID", how="left")

print("\nOutput (Dl ⨝ Dr on ID, left outer):")
print(output_df2)

print(f"\nProvenance: {prov2}")

print("\nDense tensors:")
for k, T in enumerate(prov2.to_dense()):
    side = "left (Dl)" if k == 0 else "right (Dr)"
    print(f"\n  T_{k} (output × {side}):")
    print(T)

print("\nBackward queries:")
for i in range(len(output_df2)):
    sources = prov2.backward(i)
    src_l = list(sources[0]) if len(sources[0]) > 0 else "[]"
    src_r = list(sources[1]) if len(sources[1]) > 0 else "[]"
    print(f"  Output row {i}: Dl={src_l}, Dr={src_r}")

# Sanity check: every output row has a left parent, only some have a right parent
T_l = prov2.to_dense()[0]
T_r = prov2.to_dense()[1]
print(f"\nSanity check (left outer):")
print(f"  Every row should have exactly 1 left parent:  sums = {list(T_l.sum(axis=1))}")
print(f"  Right parents only on matched rows:           sums = {list(T_r.sum(axis=1))}")
assert (T_l.sum(axis=1) == 1).all(), "Left outer should give a left parent to every output row"
print("  correct")


# ======================================================================
# TEST 3 : 1-to-N join (one left row matches several right rows)
# ======================================================================

print("\n" + "=" * 70)
print("TEST 3 : 1-to-N join (duplicate keys on right)")
print("=" * 70)
print("Test that we correctly handle the case where one Dl row produces several output rows")

A = pd.DataFrame({
    "key": [1, 2, 3],
    "val_a": ["a1", "a2", "a3"],
})
B = pd.DataFrame({
    "key": [2, 2, 3, 3, 3],
    "val_b": ["b1", "b2", "b3", "b4", "b5"],
})

print("\nA:")
print(A)
print("\nB:")
print(B)

out3, prov3 = Prov.merge(A, B, on="key", how="inner")

print("\nOutput (A ⨝ B):")
print(out3)

print(f"\nProvenance: {prov3}")

T_l = prov3.to_dense()[0]
T_r = prov3.to_dense()[1]
print("\nT_left (output × A):")
print(T_l)
print("\nT_right (output × B):")
print(T_r)

print("\nForward queries on A (where does each row of A end up?):")
for j in range(len(A)):
    dst = prov3.forward(j, dataset_index=0)
    key_val = A.iloc[j]["key"]
    print(f"  A[{j}] (key={key_val}) -> output rows {list(dst)}")

# Verify: A[1] should be in 2 output rows (the 2 matches of key=2 in B)
#         A[2] should be in 3 output rows (the 3 matches of key=3 in B)
assert len(prov3.forward(0, dataset_index=0)) == 0, "A[0] (key=1) should not match anywhere"
assert len(prov3.forward(1, dataset_index=0)) == 2, "A[1] (key=2) should match in 2 output rows"
assert len(prov3.forward(2, dataset_index=0)) == 3, "A[2] (key=3) should match in 3 output rows"
print("\n  1-to-N relationship correctly captured")


# ======================================================================
# TEST 4 : Defensive check - column name collision
# ======================================================================

print("\n" + "=" * 70)
print("TEST 4 : Defensive check on tracker column collision")
print("=" * 70)

bad_df = pd.DataFrame({
    "id": [1, 2],
    "__prov_left_row__": ["pre-existing", "data"],  # collision!
})
good_df = pd.DataFrame({"id": [1, 2], "value": ["a", "b"]})

print("\nTrying to merge a DataFrame that already has __prov_left_row__ column...")
try:
    _ = Prov.merge(bad_df, good_df, on="id")
    print(" KO: should have raised ValueError")
except ValueError as e:
    print(f" OK: ValueError correctly raised:")
    print(f"      {e}")


# ======================================================================
# TEST 5 : Verify that inputs are not mutated (defensive copy works)
# ======================================================================

print("\n" + "=" * 70)
print("TEST 5 : Inputs are not mutated by Prov.merge")
print("=" * 70)

dl_before = Dl.copy()
dr_before = Dr.copy()
_ = Prov.merge(Dl, Dr, on="ID", how="inner")
assert Dl.equals(dl_before), "Dl was mutated!"
assert Dr.equals(dr_before), "Dr was mutated!"
print(" Inputs are unchanged after Prov.merge call")
print(f" Tracker columns are NOT visible to user:  output columns = {list(output_df.columns)}")
