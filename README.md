# pandas-prov

**Row-level data provenance for pandas data preparation pipelines, using sparse binary tensors.**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Capture, query, and compose the row-level provenance (lineage) of pandas data
preparation operations with a lightweight wrapper that leaves your code almost
unchanged.

```python
from prov import Prov, ProvenanceTensor

# Replace df.query(...) by Prov.filter(...) and get the tensor for free
output_df, prov = Prov.filter(df, df["age"] >= 25)

# Trace an output row back to its sources
prov.backward(0)   # -> array of input row positions
```

Developed as a Data Quality mini-project at Université Paris Dauphine-PSL
(Executive Master IA & Data Science, supervised by K. Belhajjame).

---

## Why

When a machine-learning model surfaces a biased prediction or needs to be
audited (GDPR, AI Act, internal model governance), a central question is
*which rows of the raw dataset contributed to this particular final row?*
After a pipeline of filters, joins, SMOTE and encodings, that information is
usually lost.

**pandas-prov** solves this at row level by:

- capturing a **sparse binary tensor** T per operation, with `T[i, j] = 1`
  iff input row `j` contributed to output row `i`;
- composing a chain of such tensors into a **single end-to-end tensor** via
  boolean matrix multiplication, so that any output row can be traced back to
  the raw dataset in **one query**.

The sparse representation keeps the overhead negligible: on a 32 k-row
dataset the full-chain tensor shrinks from **1 GB (dense) to 384 KB
(sparse)** — a **2 732× compression ratio**.

---

## Supported operations

| Operation      | Category               | How provenance is captured                         |
|----------------|------------------------|----------------------------------------------------|
| `Prov.filter`     | Horizontal reduction   | Exploits pandas' preserved index labels           |
| `Prov.append`     | Vertical fusion        | Shifted-diagonal pattern per input dataset        |
| `Prov.oversample` | Horizontal augmentation (SMOTE) | Re-implements SMOTE and records BOTH parents |
| `Prov.merge`      | Relational fusion      | Injects temporary tracker columns before merge    |

Each operation returns `(output_df, ProvenanceTensor)`. The `ProvenanceTensor`
holds a list of `scipy.sparse.csr_matrix` objects (length 1 for single-input
ops, N for multi-input ops) plus pandas index metadata.

---

## Quick start

### Installation

```bash
git clone https://github.com/ChloeXP/pandas-prov.git
cd pandas-prov
pip install -r requirements.txt
```

### Minimal example

```python
import pandas as pd
from prov import Prov, ProvenanceTensor

df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "age":  [24, 17, 35, 45, 19],
})

# Capture provenance while filtering
out, p1 = Prov.filter(df, df["age"] >= 25)

# Backward query: output row 0 (Charlie) comes from input row 2
print(p1.backward(0))        # -> [array([2])]

# Chain two operations, then compose
out2, p2 = Prov.filter(out, out["age"] <= 40)
p_full = ProvenanceTensor.compose_chain([p1, p2])

# One query gives the end-to-end lineage
print(p_full.backward(0))    # -> original raw df row(s)
```

### Running the three real-world pipelines

```bash
python pipeline_compas.py    # Compass recidivism dataset
python pipeline_german.py    # German Credit dataset
python pipeline_census.py    # Census Adult dataset

# Cross-dataset summary report
python summary.py
```

Each script writes to `processed_results/<dataset>/`:
- the processed `.csv`,
- the end-to-end provenance tensor as `.npz`,
- a human-readable `provenance_report.txt`.

### Running the unit tests

```bash
python test_filter.py
python test_append.py
python test_oversample.py
python test_merge.py
python test_compose.py
```

Each test script reproduces the canonical examples from the project
specification (slides 6, 9, 10) plus structural and mathematical sanity
checks.

---

## Experimental results

Benchmarks on three fairness datasets (final composed tensor):

| Dataset | Shape            | nnz     | Dense     | Sparse (actual) | Compression |
|---------|------------------|---------|-----------|-----------------|-------------|
| Compass | (3 972, 7 214)   | 3 974   | 28.65 MB  | 35.8 KB         | **801×**    |
| German  | (1 220, 1 000)   | 1 591   | 1.22 MB   | 14.3 KB         | **85×**     |
| Census  | (32 198, 32 561) | 42 639  | 1 048 MB  | 384 KB          | **2 732×**  |

Query latency, averaged over 100-200 random samples:

| Dataset | Backward query | Forward query |
|---------|----------------|---------------|
| Compass | 0.041 ms       | 0.043 ms      |
| German  | 0.037 ms       | 0.038 ms      |
| Census  | 0.036 ms       | 0.129 ms      |

Query times are nearly constant across dataset sizes — a direct consequence
of CSR's `O(nnz_per_row)` row slicing.

---

## Architecture

```
prov.py                     # core module (ProvenanceTensor + Prov)
├── ProvenanceTensor        # holds list of sparse matrices + metadata
│   ├── .backward(i)        # output -> inputs
│   ├── .forward(j)         # input -> outputs
│   └── .compose_chain([…]) # fold a linear pipeline
└── Prov                    # namespace of @staticmethod operations
    ├── .filter(df, mask)
    ├── .append([dfs])
    ├── .oversample(df, target_col)
    └── .merge(left, right, …)
```

Key design decisions:

1. **Uniform list format** — every tensor is wrapped as a list, so query code
   works the same for single-input and multi-input operations.
2. **Sparse CSR, int8** — matches the access pattern of backward queries and
   minimizes memory.
3. **Position-based indexing** — tensors use positions (0..n-1), not pandas
   labels, so chaining is robust to non-contiguous indices.
4. **SMOTE re-implementation** — we re-wrote SMOTE from scratch to record
   both true parents of each synthetic sample. `imblearn` does not expose
   this information, and a nearest-neighbour approximation would be
   mathematically incorrect.
5. **Tracker-column injection for Merge** — delegates all join combinatorics
   to pandas, works uniformly for inner/left/right/full outer joins and for
   1-to-N / N-to-N keys.

See the full report in [`docs/`](./docs/) for the detailed rationale.

---

## Pipeline composition

For a linear pipeline `D₀ → [op₁] → D₁ → [op₂] → D₂` with tensors
`T₁` and `T₂`, the end-to-end provenance is given by **boolean matrix
multiplication**:

```
T_composed = sign(T₂ · T₁)
```

This is efficient (SciPy's native sparse matmul), associative (so
`compose_chain` is a fold-left), and preserves sparsity.

Multi-input operations (`merge`, `append`) upstream of a single-input op are
composed branch-by-branch, preserving the multi-input structure through the
chain.

---

## Limitations

- Does **not** cover value-transformation operations (imputation,
  normalization, one-hot encoding). These preserve row identity, so our
  framework stays consistent, but column-level provenance would need a 4-th
  order tensor.
- Pipeline composition supports **linear chains only**. General DAG pipelines
  (e.g. two parallel filters that then merge) require a graph-aware
  `compose_graph` that is not implemented.
- SMOTE's brute-force k-NN is `O(m² · d)`. On the Census dataset's ~5700
  minority samples this takes ~4 s. Swapping in `sklearn.neighbors.BallTree`
  would bring this down to `O(m log m · d)`.

---

## Datasets

The three datasets shipped under `Datasets/` are the standard fairness ML
benchmarks:

- **Compass**: Pro Publica recidivism dataset (7 214 rows × 53 cols)
- **German Credit**: UCI German Credit dataset (1 000 rows × 21 cols)
- **Census Adult**: UCI Adult income dataset (32 561 rows × 15 cols)

All three are redistributed here under their respective original licenses for
reproducibility.

---

## Project layout

```
pandas-prov/
├── prov.py                        # core module
├── test_filter.py                 # unit tests for each operation
├── test_append.py
├── test_oversample.py
├── test_merge.py
├── test_compose.py                # Task 2 — pipeline composition tests
├── pipeline_compas.py             # end-to-end pipelines on real data
├── pipeline_german.py
├── pipeline_census.py
├── summary.py                     # cross-dataset summary generator
├── Datasets/                      # raw input data
│   ├── compas.csv
│   ├── german.csv
│   └── census.csv
├── processed_results/             # pre-computed outputs (reproducible)
│   ├── summary.txt
│   ├── compas/
│   ├── german/
│   └── census/
├── requirements.txt
├── README.md
└── LICENSE
```

---


## License

MIT. See [LICENSE](LICENSE) for details.

The three bundled datasets retain their original licenses:
- Compass: public domain (Pro Publica)
- German Credit: UCI ML Repository terms
- Census Adult: UCI ML Repository terms
