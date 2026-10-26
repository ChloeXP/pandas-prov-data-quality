"""
================================================================================
 Data Quality Mini-Project — Provenance Tracking via Sparse Binary Tensors
================================================================================

Course        : Master Exécutive IA & Data Science, Université Paris Dauphine-PSL
Author        : Xuan PENG

--------------------------------------------------------------------------------
 OBJECTIVE
--------------------------------------------------------------------------------

The goal of this project (Task 1) is to capture the data provenance of pandas
data preparation operations. Given an operation (e.g. filter, merge), its input
dataframe(s), and its output dataframe, I build a binary sparse tensor T such
that:

    T[i, j] = 1   <=>   row j of an input dataframe contributed to row i
                        of the output dataframe.

The four operations covered in the mini-project are:

    Operation       Category                Inputs   Tensor structure
    -----------     ---------------------   ------   ------------------------
    filter          Horizontal reduction    1        single matrix (n_out, n_in)
    oversample      Horizontal augmentation 1        single matrix (n_out, n_in)
    append          Data fusion (vertical)  N        list of N matrices
    merge           Data fusion (relational) 2       pair of 2 matrices

--------------------------------------------------------------------------------
 DESIGN DECISIONS
--------------------------------------------------------------------------------

(1) Uniform list-based tensor format
    Every operation stores its provenance as a list of csr_matrix objects, with
    length equal to the number of input datasets. Operations with a single
    input (filter, oversample) therefore use a list of length 1. This avoids
    type-dispatching ("is it a matrix or a tuple?") in the query code: every
    method iterates uniformly over the list.

(2) Sparse representation (scipy.sparse.csr_matrix, int8)
    Provenance matrices are extremely sparse in practice. For example, a
    filter that keeps 5% of a 1M-row dataset produces a matrix with 50k
    non-zero cells out of 50 billion, a sparsity above 99.999%. I use
    CSR (Compressed Sparse Row) because it provides O(nnz_per_row) row
    slicing — exactly what backward queries need.

(3) Object-oriented encapsulation (ProvenanceTensor class)
    Rather than returning bare scipy matrices, I wrap them in a
    ProvenanceTensor object that also carries metadata (input/output pandas
    indices, operation type). This (a) hides the choice of sparse backend
    behind a stable API, (b) supports label-based queries, and (c) prepares
    the ground for Task 2 (pipeline composition).

(4) Decorator-style entry points (Prov class)
    The Prov class is a namespace of @staticmethod functions, one per
    operation. Each method runs the underlying pandas operation and, in
    parallel, builds the corresponding provenance tensor. The user can
    therefore continue to write near-pandas code (e.g. Prov.filter(df, cond))
    and receive both the result and its provenance in a single call.

(5) Provenance capture strategies (operation-specific)

    - filter: pandas preserves the original index labels of kept rows in
      the output dataframe. I exploit that fact: we look up the positions
      of the kept rows in the original frame and place a '1' in each
      corresponding column of the tensor.

    - oversample: rather than calling imblearn (whose API does not expose
      the two parents of each synthetic sample), I re-implement the SMOTE
      algorithm. For each synthetic row, we record BOTH parents that were
      interpolated between, yielding exactly two '1's per synthetic row of
      the tensor. This gives the mathematically faithful provenance.

    - append: pd.concat stacks dataframes vertically, so the i-th output
      row of dataset k sits at position (offset_k + i) in the result. I
      build one tensor per input dataset, each with a shifted-diagonal
      pattern of '1's.

    - merge: pandas does NOT directly expose which (left_row, right_row)
      pair produced each output row, especially under 1-to-N or N-to-N
      joins. I solve this by injecting a "tracker" column into each input
      (containing the positional row index), letting pandas propagate it
      through the merge, and reading the trackers back from the output.
      The technique transparently handles all join types (inner, left,
      right, full outer) thanks to NaN-handling on unmatched rows.

================================================================================
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# ==============================================================================
# 1. ProvenanceTensor — the abstraction returned by every Prov operation
# ==============================================================================

class ProvenanceTensor:
    """
    Encapsulates one or more sparse binary tensors capturing the provenance
    of a single data preparation operation.

    The tensors are always stored as a *list*, with one element per input
    dataset. This uniform structure means that the same query code works
    regardless of whether the underlying operation took one input (filter,
    oversample) or several (append, merge).

    Attributes
    ----------
    tensors : list[scipy.sparse.csr_matrix]
        One sparse binary matrix per input dataset.
        tensors[k][i, j] == 1  ==>  row j of input dataset k contributed
                                    to row i of the output dataset.
    op_type : str
        Name of the operation ("filter", "append", "oversample", "merge").
    input_indices : list[pd.Index]
        Original pandas indices of each input dataframe (preserved for
        label-based queries).
    output_index : pd.Index
        Pandas index of the output dataframe.

    Note on duality of formats:
        For Append and Merge, both tensors have the same number of rows
        (= n_output). The semantic difference is captured by the *pattern*
        of non-zero entries:
          - Append : each output row has a '1' in *exactly one* tensor
                     (disjunctive: comes from one side or the other).
          - Merge  : each output row has a '1' in *both* tensors simultaneously
                     (conjunctive: combines one row from each side).
        For outer joins, Merge degrades gracefully toward Append behavior
        on unmatched rows.
    """

    def __init__(self, tensors, op_type, input_indices, output_index):
        # Always normalize to a list of tensors, even for single-input ops.
        # This is the cornerstone of the uniform architecture.
        if not isinstance(tensors, (list, tuple)):
            tensors = [tensors]
        self.tensors = list(tensors)
        self.op_type = op_type
        self.input_indices = list(input_indices)
        self.output_index = output_index

    # ------------------------------------------------------------------
    # Convenience properties
    # (Exposed as @property to keep the API uniform: e.g. prov.shape
    # behaves like a plain attribute, even though it is computed.)
    # ------------------------------------------------------------------

    @property
    def n_inputs(self):
        """Number of input datasets for this operation."""
        return len(self.tensors)

    @property
    def shape(self):
        """List of shapes, one per input dataset."""
        return [t.shape for t in self.tensors]

    @property
    def nnz(self):
        """Total number of non-zero entries across all tensors.
        Used as a primary indicator of storage efficiency."""
        return sum(t.nnz for t in self.tensors)

    @property
    def tensor(self):
        """
        Convenience shortcut for single-input operations.
        Returns the unique matrix when len(self.tensors) == 1.
        Raises ValueError otherwise.
        """
        if len(self.tensors) != 1:
            raise ValueError(
                f"This operation has {len(self.tensors)} input datasets; "
                f"use .tensors[k] to access them individually."
            )
        return self.tensors[0]

    # ------------------------------------------------------------------
    # Provenance queries (Task 2 — backward and forward lookups)
    # ------------------------------------------------------------------

    def backward(self, output_pos):
        """
        Backward query: which input rows produced this output row?

        Returns a list with one entry per input dataset. For a typical
        merge, both entries will be non-empty (conjunctive sources). For
        a filter or append, only one entry will contain a position.

        Parameters
        ----------
        output_pos : int
            Positional index (0..n_out-1) of an output row.

        Returns
        -------
        list[np.ndarray of int]
            Per-dataset positional indices of contributing input rows.
        """
        # CSR row slicing is O(nnz_per_row), making backward queries
        # essentially free on sparse tensors.
        return [t.getrow(output_pos).nonzero()[1] for t in self.tensors]

    def forward(self, input_pos, dataset_index=0):
        """
        Forward query: where does this input row end up in the output?

        Parameters
        ----------
        input_pos : int
            Positional index of an input row (within its dataset).
        dataset_index : int, default 0
            Which input dataset the row belongs to. Mandatory for
            multi-input operations (append, merge).

        Returns
        -------
        np.ndarray of int
            Positional indices of output rows derived (in part) from
            this input row. Empty array if the row was filtered out
            or unmatched.
        """
        if not (0 <= dataset_index < self.n_inputs):
            raise IndexError(
                f"dataset_index={dataset_index} out of range "
                f"(this op has {self.n_inputs} input datasets)."
            )
        # Internally, getcol triggers a CSR->CSC conversion on demand.
        # For very large tensors this could be cached if many forward
        # queries are issued back-to-back.
        return self.tensors[dataset_index].getcol(input_pos).nonzero()[0]

    # ------------------------------------------------------------------
    # Debugging helpers
    # ------------------------------------------------------------------

    def __repr__(self):
        shapes = ", ".join(str(s) for s in self.shape)
        return (f"ProvenanceTensor(op={self.op_type!r}, "
                f"n_inputs={self.n_inputs}, shapes=[{shapes}], "
                f"nnz={self.nnz})")

    def to_dense(self):
        """
        Return the tensors as dense ndarrays.
        Intended for small examples only — calling this on a 1M-row
        tensor would defeat the purpose of sparse storage.
        """
        return [t.toarray() for t in self.tensors]

    # ==================================================================
    # Task 2 — Pipeline composition (boolean matrix multiplication)
    # ==================================================================
    #
    # If D0 -> [op1] -> D1 -> [op2] -> D2 with local tensors T1 (n1, n0)
    # and T2 (n2, n1), then the composed tensor T_compo (n2, n0) is given
    # by the BOOLEAN matrix product:
    #
    #     T_compo = sign(T2 @ T1)
    #
    # Proof:
    #     (T2 @ T1)[i, j] = sum_k  T2[i, k] * T1[k, j]
    # counts the number of intermediate rows k in D1 that link j (in D0)
    # to i (in D2). For provenance we only care whether AT LEAST ONE such
    # k exists, so we binarize the result.
    #
    # Multi-input operations (merge, append) generate a LIST of tensors;
    # composing a downstream single-input op with them simply composes
    # branch-by-branch and returns a new list of the same length.

    @staticmethod
    def _bool_matmul(T_left, T_right):
        """
        Boolean matrix product of two sparse binary matrices.

        Computes T_left @ T_right and binarizes the result by setting
        all non-zero entries to 1. This is the algebraic core of
        pipeline composition.

        Parameters
        ----------
        T_left, T_right : scipy.sparse.csr_matrix
            Must have compatible shapes: T_left.shape[1] == T_right.shape[0].

        Returns
        -------
        scipy.sparse.csr_matrix
            Boolean product, with all data entries set to int8 '1'.
        """
        product = (T_left @ T_right).tocsr()
        # Binarization: drop the actual counts, keep only "exists or not".
        # We rebuild a clean csr_matrix to avoid dtype/canonical-form issues.
        if product.nnz > 0:
            product.data = np.ones_like(product.data, dtype=np.int8)
        return product

    @staticmethod
    def compose(prov_upstream, prov_downstream):
        """
        Compose two consecutive ProvenanceTensors in a pipeline.

        Given:
            D_in -> [op_upstream] -> D_mid -> [op_downstream] -> D_out
        with provenance tensors prov_upstream (for op_upstream) and
        prov_downstream (for op_downstream), returns a ProvenanceTensor
        describing the direct relationship D_in -> D_out.

        Supported cases:
          - upstream is single-input (filter, oversample),
            downstream is single-input  ->  scalar composition.
          - upstream is multi-input (merge, append),
            downstream is single-input  ->  branch-by-branch composition;
            the result has the same number of input datasets as upstream.

        Not supported here (yet):
          - downstream is multi-input. A multi-input downstream operation
            has its own native tensor structure (one per input dataset),
            so composing it would require linking each of its inputs to
            possibly different upstream pipelines. This is a DAG-level
            problem, not a chain-level one, and is outside the scope of
            compose() / compose_chain().

        Parameters
        ----------
        prov_upstream : ProvenanceTensor
            Provenance tensor of the upstream operation.
        prov_downstream : ProvenanceTensor
            Provenance tensor of the downstream operation. Must satisfy:
            prov_downstream.tensors[0].shape[1] == n_mid  (rows of D_mid).

        Returns
        -------
        ProvenanceTensor
            Composed provenance, op_type marked as "composed".
        """
        # We currently support a single-input downstream operation only.
        if prov_downstream.n_inputs != 1:
            raise NotImplementedError(
                f"compose() does not yet support a multi-input downstream "
                f"operation (got n_inputs={prov_downstream.n_inputs}). "
                f"Such pipelines form a DAG rather than a linear chain."
            )

        T_down = prov_downstream.tensors[0]   # shape (n_out, n_mid)
        n_out = T_down.shape[0]
        n_mid = T_down.shape[1]

        # Sanity check on dimension compatibility:
        # every upstream tensor has n_mid rows (the size of the intermediate dataset).
        for k, T_up in enumerate(prov_upstream.tensors):
            if T_up.shape[0] != n_mid:
                raise ValueError(
                    f"Dimension mismatch on branch {k}: upstream tensor has "
                    f"shape {T_up.shape}, but downstream expects {n_mid} mid-rows."
                )

        # Branch-by-branch boolean matrix product.
        composed_tensors = [
            ProvenanceTensor._bool_matmul(T_down, T_up)
            for T_up in prov_upstream.tensors
        ]

        return ProvenanceTensor(
            tensors=composed_tensors,
            op_type="composed",
            input_indices=prov_upstream.input_indices,   # same original inputs
            output_index=prov_downstream.output_index,   # same final output
        )

    @staticmethod
    def compose_chain(provs):
        """
        Compose a linear chain of ProvenanceTensors.

        Given a list [T_1, T_2, ..., T_n] corresponding to consecutive
        operations in a pipeline:
            D_0 -> [op_1] -> D_1 -> [op_2] -> ... -> D_n
        returns a single ProvenanceTensor that directly relates D_0 to D_n.

        The composition is performed left-to-right (fold/reduce):
            result = compose(compose(compose(T_1, T_2), T_3), T_4) ...

        Note on multi-input operations in the chain:
            If T_1 is multi-input (merge/append) and the rest are single-input,
            the multi-input structure is preserved throughout. If a later
            operation is multi-input, compose() will raise NotImplementedError.

        Parameters
        ----------
        provs : list[ProvenanceTensor]
            Chain of provenance tensors, in order from earliest to latest.

        Returns
        -------
        ProvenanceTensor
            Composed provenance describing D_0 -> D_n.
        """
        if not provs:
            raise ValueError("compose_chain() requires at least one tensor.")
        if len(provs) == 1:
            # Nothing to compose; return as-is (could also return a copy).
            return provs[0]

        # Fold-left: result <- compose(result, next)
        result = provs[0]
        for next_prov in provs[1:]:
            result = ProvenanceTensor.compose(result, next_prov)
        return result


# ==============================================================================
# 2. Prov — factory of provenance-tracking pandas operations
# ==============================================================================

class Prov:
    """
    Namespace of pandas operations wrapped to also produce a
    ProvenanceTensor describing the input -> output relationship.

    Every method is a @staticmethod: the class itself carries no state,
    and serves only to group the operations logically under a common name.
    Calling pattern:

        output_df, prov = Prov.<operation>(df, ...)
    """

    # ============================================================
    # 2.1  Filter — Horizontal Reduction
    # ============================================================

    @staticmethod
    def filter(df, mask):
        """
        Filter rows of a DataFrame and capture the row-level provenance.

        Two equivalent input forms are accepted:
          - a boolean mask of length len(df), e.g. df['Age'] >= 25
          - a query string, e.g. "Age >= 25"

        Provenance characterization:
          Each output row has exactly one parent (the corresponding input
          row that satisfied the condition). The tensor therefore has
          exactly one '1' per row, totalling n_output non-zero entries.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        mask : array-like of bool, or str
            Selection criterion.

        Returns
        -------
        output_df : pd.DataFrame
            Filtered dataframe (original pandas index preserved).
        prov : ProvenanceTensor
            Provenance tensor of shape (n_kept, n_input).
        """
        n_input = len(df)

        # Step 1: identify which positional rows are kept
        # I work with positions (0..n-1) rather than pandas index labels
        # to make the tensor robust to non-trivial indices (e.g. a df that
        # already comes from a previous filter has index labels like [2,3]).
        if isinstance(mask, str):
            kept_df = df.query(mask)
            # `get_indexer` translates labels of the kept rows back into
            # the positional space of the original df.
            kept_positions = df.index.get_indexer(kept_df.index)
        else:
            mask = np.asarray(mask, dtype=bool)
            if len(mask) != n_input:
                raise ValueError(
                    f"Boolean mask length {len(mask)} "
                    f"does not match df length {n_input}"
                )
            kept_positions = np.flatnonzero(mask)
            kept_df = df.iloc[kept_positions]

        n_output = len(kept_positions)

        # Step 2: build the sparse tensor in COO triplet form
        # Each output row i has a single '1' at column kept_positions[i].
        rows = np.arange(n_output)
        cols = kept_positions
        data = np.ones(n_output, dtype=np.int8)

        tensor = csr_matrix(
            (data, (rows, cols)),
            shape=(n_output, n_input),
        )

        prov = ProvenanceTensor(
            tensors=[tensor],
            op_type="filter",
            input_indices=[df.index.copy()],
            output_index=kept_df.index.copy(),
        )

        return kept_df, prov

    # ============================================================
    # 2.2  Append — Data Fusion (vertical concatenation)
    # ============================================================

    @staticmethod
    def append(dfs, **concat_kwargs):
        """
        Vertically concatenate a list of DataFrames, with provenance.

        Provenance characterization:
          Each output row comes from EXACTLY ONE input dataset, at the
          corresponding position. The result is a list of N tensors, each
          containing a "shifted-diagonal" of '1's: the k-th tensor has its
          diagonal of ones shifted by the cumulative size of all previous
          datasets.

        Parameters
        ----------
        dfs : list[pd.DataFrame]
            Input dataframes to stack vertically. Schemas may differ;
            mismatched columns are filled with NaN by pandas.
        **concat_kwargs :
            Forwarded to pd.concat (e.g. ignore_index=True).

        Returns
        -------
        output_df : pd.DataFrame
            Concatenated dataframe.
        prov : ProvenanceTensor
            List of N tensors; the k-th has shape (n_output, len(dfs[k])).
        """
        if not isinstance(dfs, (list, tuple)) or len(dfs) == 0:
            raise ValueError("`dfs` must be a non-empty list of DataFrames.")

        # Step 1: run the actual concat
        output_df = pd.concat(dfs, axis=0, **concat_kwargs)
        n_output = len(output_df)

        # Step 2: build one shifted-diagonal tensor per input
        # The k-th dataset occupies output rows [offset_k, offset_k + n_k).
        tensors = []
        offset = 0
        for df_k in dfs:
            n_k = len(df_k)
            # Output rows [offset, offset+n_k) come from input rows [0, n_k)
            rows = np.arange(offset, offset + n_k, dtype=np.int32)
            cols = np.arange(n_k, dtype=np.int32)
            data = np.ones(n_k, dtype=np.int8)
            tensor_k = csr_matrix(
                (data, (rows, cols)),
                shape=(n_output, n_k),
            )
            tensors.append(tensor_k)
            offset += n_k

        prov = ProvenanceTensor(
            tensors=tensors,
            op_type="append",
            input_indices=[df_k.index.copy() for df_k in dfs],
            output_index=output_df.index.copy(),
        )

        return output_df, prov

    # ============================================================
    # 2.3  Oversample — Horizontal Augmentation (SMOTE)
    # ============================================================
    #
    # Design note: rather than relying on imblearn.over_sampling.SMOTE
    # (whose API hides the parent pairs), I re-implement SMOTE here so
    # that the two parents of each synthetic sample are explicitly
    # tracked. This gives the FAITHFUL provenance — every synthetic
    # row of the output is shown to derive from exactly the two
    # original rows that were interpolated. An a-posteriori "nearest
    # neighbour" approximation would have been simpler but incorrect.
    # ============================================================

    @staticmethod
    def oversample(df, target_col, k_neighbors=5, random_state=None):
        """
        SMOTE-style oversampling with exact provenance tracking.

        The standard SMOTE algorithm proceeds as follows. For each
        minority-class sample x_i, I find its k nearest minority
        neighbours, pick one of them at random (call it x_j), and
        create a synthetic sample by linear interpolation:

              x_synth = x_i + lambda * (x_j - x_i),     lambda ~ U(0, 1)

        I repeat this until both classes have equal size. Each synthetic
        sample is therefore born of EXACTLY two parents — and our
        provenance tensor encodes this directly with two '1's per
        synthetic row.

        Provenance characterization:
          - Original rows 0..n_input-1 : identity (one '1' on the diagonal)
          - Synthetic rows             : two '1's at the columns of their
                                         two parents

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe. All columns except `target_col` are treated
            as numeric features.
        target_col : str
            Name of the binary target column.
        k_neighbors : int, default 5
            Number of nearest neighbours considered for interpolation.
        random_state : int or None
            Seed for reproducibility.

        Returns
        -------
        output_df : pd.DataFrame
            Concatenation of (original rows) and (synthetic rows).
            Index is reset to 0..n_out-1.
        prov : ProvenanceTensor
            Single tensor of shape (n_output, n_input) with exactly
            n_input + 2 * n_synthetic non-zero entries.
        """
        rng = np.random.default_rng(random_state)
        n_input = len(df)

        # Step 1: split features and target
        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols].to_numpy(dtype=float)
        y = df[target_col].to_numpy()

        # Step 2: identify the minority class
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            raise ValueError(
                f"Oversample expects a binary target, got {len(classes)} classes."
            )
        minority_class = classes[np.argmin(counts)]
        n_minority = counts.min()
        n_majority = counts.max()
        n_to_generate = n_majority - n_minority

        # Positional indices (in the original df) of minority samples.
        minority_positions = np.flatnonzero(y == minority_class)
        X_minority = X[minority_positions]

        # Step 3: compute k-NN graph within the minority class
        # We use brute-force pairwise squared distances; the minority
        # class is small by definition, so this remains cheap.
        k_eff = min(k_neighbors, len(minority_positions) - 1)
        if k_eff < 1:
            raise ValueError(
                "Not enough minority samples to perform oversampling "
                f"(found {len(minority_positions)})."
            )

        diffs = X_minority[:, None, :] - X_minority[None, :, :]
        dists = (diffs ** 2).sum(axis=-1)
        # Each sample should not be its own neighbour.
        np.fill_diagonal(dists, np.inf)
        # Indices, in the minority-only space, of each sample's k nearest.
        nn_in_minority = np.argsort(dists, axis=1)[:, :k_eff]

        # Step 4: generate synthetic samples
        # For each new sample we record (parent1, parent2, lambda) so that
        # the provenance is recoverable from the tensor alone.
        synthetic_features = np.empty((n_to_generate, X.shape[1]))
        parent1_positions = np.empty(n_to_generate, dtype=np.int32)
        parent2_positions = np.empty(n_to_generate, dtype=np.int32)

        for i in range(n_to_generate):
            # Round-robin selection of parent 1 ensures every minority
            # sample contributes roughly equally.
            p1_local = i % len(minority_positions)
            # Parent 2 is sampled uniformly from p1's k nearest minorities.
            p2_local = rng.choice(nn_in_minority[p1_local])
            lam = rng.random()

            x1 = X_minority[p1_local]
            x2 = X_minority[p2_local]
            synthetic_features[i] = x1 + lam * (x2 - x1)

            # Map back from minority-local index to original-df position.
            parent1_positions[i] = minority_positions[p1_local]
            parent2_positions[i] = minority_positions[p2_local]

        # Step 5: assemble output dataframe
        # The conventional layout (originals first, synthetics afterwards)
        # matches imblearn's behavior.
        synthetic_df = pd.DataFrame(synthetic_features, columns=feature_cols)
        synthetic_df[target_col] = minority_class
        synthetic_df = synthetic_df[df.columns]  # preserve column order

        output_df = pd.concat(
            [df.reset_index(drop=True), synthetic_df],
            axis=0,
            ignore_index=True,
        )
        n_output = len(output_df)

        # Step 6: build the provenance tensor
        # I assemble it in a single COO construction with three blocks:
        #   - diagonal block for originals  : n_input ones
        #   - parent-1 contributions        : n_synth ones
        #   - parent-2 contributions        : n_synth ones
        rows = np.concatenate([
            np.arange(n_input, dtype=np.int32),
            np.arange(n_input, n_output, dtype=np.int32),
            np.arange(n_input, n_output, dtype=np.int32),
        ])
        cols = np.concatenate([
            np.arange(n_input, dtype=np.int32),
            parent1_positions,
            parent2_positions,
        ])
        data = np.ones(len(rows), dtype=np.int8)

        tensor = csr_matrix(
            (data, (rows, cols)),
            shape=(n_output, n_input),
        )

        prov = ProvenanceTensor(
            tensors=[tensor],
            op_type="oversample",
            input_indices=[df.index.copy()],
            output_index=output_df.index.copy(),
        )

        return output_df, prov

    # ============================================================
    # 2.4  Merge — Data Fusion (relational join)
    # ============================================================
    #
    # Design note: capturing the provenance of a relational join is
    # non-trivial because pandas does not directly tell us which
    # (left_row, right_row) pair produced each output row. A naive
    # a-posteriori matching by join key would fail under 1-to-N and
    # N-to-N joins (which left row produced which output row?).
    #
    # I solve this elegantly by INJECTING tracker columns into the
    # input dataframes before the merge. These columns simply contain
    # the positional row index; pandas propagates them through the
    # join, after which I read them back to identify the parent of
    # each output row. The technique handles all join types uniformly
    # (inner, left, right, full outer) thanks to NaN-handling for
    # unmatched rows.
    # ============================================================

    # Tracker column names. Chosen to minimize collision risk; I still
    # validate that they are absent from the user's inputs to avoid
    # silent bugs.
    _PROV_COL_L = "__prov_left_row__"
    _PROV_COL_R = "__prov_right_row__"

    @staticmethod
    def merge(left, right, **merge_kwargs):
        """
        Merge two DataFrames and capture the row-level provenance.

        Provenance characterization:
          For inner joins, each output row has EXACTLY one '1' in each of
          the two tensors (conjunctive: combines one row from each side).
          For outer joins, unmatched rows have all-zero rows in the side
          they did not match — semantically equivalent to an Append
          contribution from a single side.

        Parameters
        ----------
        left, right : pd.DataFrame
            Input dataframes.
        **merge_kwargs :
            Forwarded to pd.merge (on, left_on, right_on, how, suffixes...).

        Returns
        -------
        output_df : pd.DataFrame
            Merged dataframe (index reset to 0..n_out-1).
        prov : ProvenanceTensor
            List of exactly 2 tensors:
              - tensors[0]: shape (n_output, len(left))
              - tensors[1]: shape (n_output, len(right))
        """
        # Step 1: defensive collision check
        # Without this guard, a user dataframe that happens to already
        # contain a column named __prov_left_row__ would silently
        # collide with our tracker — producing a corrupt tensor.
        for col in (Prov._PROV_COL_L, Prov._PROV_COL_R):
            if col in left.columns:
                raise ValueError(
                    f"Input `left` already contains a column named {col!r}. "
                    f"Please rename it before calling Prov.merge."
                )
            if col in right.columns:
                raise ValueError(
                    f"Input `right` already contains a column named {col!r}. "
                    f"Please rename it before calling Prov.merge."
                )

        n_left = len(left)
        n_right = len(right)

        # Step 2: inject tracker columns
        # We copy the inputs to keep them pristine for the user.
        left_tracked = left.copy()
        right_tracked = right.copy()
        left_tracked[Prov._PROV_COL_L] = np.arange(n_left, dtype=np.int64)
        right_tracked[Prov._PROV_COL_R] = np.arange(n_right, dtype=np.int64)

        # Step 3: run the actual merge
        # Pandas automatically propagates the tracker columns into the
        # merged result. For unmatched rows in an outer join, the tracker
        # of the non-matched side becomes NaN.
        merged = pd.merge(left_tracked, right_tracked, **merge_kwargs)
        n_output = len(merged)

        # Step 4: build T_left from the left tracker column
        # Skip rows where the left side did not match (NaN tracker).
        prov_l_col = merged[Prov._PROV_COL_L]
        valid_l = prov_l_col.notna()
        rows_l = np.flatnonzero(valid_l.values).astype(np.int32)
        cols_l = prov_l_col[valid_l].to_numpy(dtype=np.int32)
        data_l = np.ones(len(rows_l), dtype=np.int8)
        T_left = csr_matrix(
            (data_l, (rows_l, cols_l)),
            shape=(n_output, n_left),
        )

        # Step 5: build T_right symmetrically
        prov_r_col = merged[Prov._PROV_COL_R]
        valid_r = prov_r_col.notna()
        rows_r = np.flatnonzero(valid_r.values).astype(np.int32)
        cols_r = prov_r_col[valid_r].to_numpy(dtype=np.int32)
        data_r = np.ones(len(rows_r), dtype=np.int8)
        T_right = csr_matrix(
            (data_r, (rows_r, cols_r)),
            shape=(n_output, n_right),
        )

        # Step 6: clean up the output
        # The user receives a perfectly normal merged dataframe;
        # tracker columns disappear from view.
        output_df = merged.drop(
            columns=[Prov._PROV_COL_L, Prov._PROV_COL_R]
        ).reset_index(drop=True)

        prov = ProvenanceTensor(
            tensors=[T_left, T_right],
            op_type="merge",
            input_indices=[left.index.copy(), right.index.copy()],
            output_index=output_df.index.copy(),
        )

        return output_df, prov
