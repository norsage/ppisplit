"""Fold assignment: k-fold (greedy balanced) and Monte Carlo random splits."""

from __future__ import annotations

import heapq

import numpy as np
import pandas as pd


def assign_kfold(
    complex_clusters: dict[int, list[str]],
    n_folds: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Greedy min-heap k-fold balancing on cluster sizes.

    Each cluster is placed atomically into the currently smallest fold.
    Returns a wide-format DataFrame with columns ``uid, split_0, ..., split_{n_folds-1}``,
    values ``train`` or ``val``.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    rng = np.random.RandomState(seed)
    items = list(complex_clusters.items())
    rng.shuffle(items)
    items.sort(key=lambda kv: len(kv[1]), reverse=True)

    heap: list[tuple[int, int]] = [(0, i) for i in range(n_folds)]
    heapq.heapify(heap)

    cluster_to_fold: dict[int, int] = {}
    for cid, uids in items:
        fold_size, fold_id = heapq.heappop(heap)
        cluster_to_fold[cid] = fold_id
        heapq.heappush(heap, (fold_size + len(uids), fold_id))

    rows: list[dict[str, str]] = []
    for cid, uids in complex_clusters.items():
        fold_id = cluster_to_fold[cid]
        for uid in uids:
            row: dict[str, str] = {"uid": uid}
            for f in range(n_folds):
                row[f"split_{f}"] = "val" if f == fold_id else "train"
            rows.append(row)

    return pd.DataFrame(rows).sort_values("uid").reset_index(drop=True)


def assign_montecarlo(
    complex_clusters: dict[int, list[str]],
    n_splits: int,
    val_ratio: float = 0.25,
    seed: int = 42,
) -> pd.DataFrame:
    """Monte Carlo random splits at the cluster level.

    Each split independently assigns whole clusters to val until ``val_ratio``
    of UIDs are covered. Returns ``uid, split_0, ..., split_{n_splits-1}``.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    cluster_ids = sorted(complex_clusters.keys())
    cluster_sizes = {cid: len(uids) for cid, uids in complex_clusters.items()}
    total = sum(cluster_sizes.values())
    target_val = int(total * val_ratio)

    all_uids: list[str] = []
    uid_to_cluster: dict[str, int] = {}
    for cid, uids in complex_clusters.items():
        for uid in uids:
            all_uids.append(uid)
            uid_to_cluster[uid] = cid
    all_uids.sort()

    result = pd.DataFrame({"uid": all_uids})
    for split_idx in range(n_splits):
        rng = np.random.RandomState(seed + split_idx)
        shuffled = rng.permutation(cluster_ids)

        val_clusters: set[int] = set()
        val_count = 0
        for cid in shuffled:
            if val_count >= target_val:
                break
            val_clusters.add(int(cid))
            val_count += cluster_sizes[int(cid)]

        result[f"split_{split_idx}"] = [
            "val" if uid_to_cluster[uid] in val_clusters else "train"
            for uid in all_uids
        ]

    return result
