"""
Deduplication of model-input feature vectors (Reviewer 1, Comment 8).

Duplicates are defined on the exact 33-float model input produced by
feature_mapper.map_cicids_to_features — the representation the classifier sees.
Vectors that occur with more than one label have ambiguous ground truth and are
handled by an explicit policy:
  - "drop"     : remove every row of a conflicting vector (default, v2 protocol)
  - "majority" : keep the vector with its most frequent label; ties are dropped
Deterministic: the first occurrence (lowest row index) is kept.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from edge.training.label_mapper import CLASS_NAMES

CONFLICT_POLICIES = ("drop", "majority")


def row_hashes(X: np.ndarray) -> np.ndarray:
    """64-bit hash per row of X (exact float equality)."""
    return pd.util.hash_pandas_object(pd.DataFrame(X), index=False).to_numpy()


def _counts(y: np.ndarray) -> Dict[str, int]:
    return {CLASS_NAMES[k]: int(np.sum(y == k)) for k in range(len(CLASS_NAMES))}


def deduplicate(
    X: np.ndarray,
    y: np.ndarray,
    conflict_policy: str = "drop",
) -> Tuple[np.ndarray, Dict]:
    """
    Return (keep_indices, report). keep_indices are sorted row indices into X/y.
    """
    if conflict_policy not in CONFLICT_POLICIES:
        raise ValueError(f"conflict_policy must be one of {CONFLICT_POLICIES}")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    h = row_hashes(X)
    df = pd.DataFrame({"h": h, "y": y, "i": np.arange(len(y))})
    n_labels = df.groupby("h")["y"].nunique()
    conflict_h = n_labels.index[n_labels > 1]
    is_conflict = df["h"].isin(conflict_h).to_numpy()

    clean = df[~is_conflict].drop_duplicates("h", keep="first")
    kept = [clean["i"].to_numpy()]

    resolved = 0
    if conflict_policy == "majority" and len(conflict_h):
        c = df[is_conflict]
        cnt = c.groupby(["h", "y"]).size().reset_index(name="n")
        top_n = cnt.groupby("h")["n"].transform("max")
        leaders = cnt[cnt["n"] == top_n]
        n_leaders = leaders.groupby("h")["y"].transform("size")
        winners = leaders[n_leaders == 1][["h", "y"]]          # unique majority label; ties dropped
        first = c.merge(winners, on=["h", "y"]).sort_values("i").drop_duplicates("h", keep="first")
        kept.append(first["i"].to_numpy())
        resolved = int(len(first))

    keep = np.sort(np.concatenate(kept))
    report = {
        "definition": "exact equality of the 33-float model-input vector",
        "conflict_policy": conflict_policy,
        "rows_in": int(len(y)),
        "rows_out": int(len(keep)),
        "redundant_copies_removed": int(len(y) - len(keep) - (int(is_conflict.sum()) - resolved)),
        "conflicting_vectors": int(len(conflict_h)),
        "rows_in_conflicting_vectors": int(is_conflict.sum()),
        "conflicting_vectors_resolved_by_majority": resolved,
        "class_counts_in": _counts(y),
        "class_counts_out": _counts(y[keep]),
        "class_counts_conflict_rows": _counts(y[is_conflict]),
    }
    return keep, report


def cross_partition_overlap(X: np.ndarray, src_idx: np.ndarray, dst_idx: np.ndarray) -> int:
    """Number of rows in dst_idx whose exact vector also appears in src_idx."""
    h = row_hashes(X)
    return int(np.isin(h[dst_idx], np.unique(h[src_idx])).sum())
