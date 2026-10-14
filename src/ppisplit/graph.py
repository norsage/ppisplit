"""Graph operations: connected components and complex-level union-find.

**Chain clusters are never used as a similarity proxy.** Complex adjacency is
built from the *direct* chain edges (:func:`complex_clusters_from_edges`).

The removed predecessor, ``complex_clusters_from_chain_clusters``, tested
whether two complexes' chain *clusters* overlapped. Cluster co-membership is
the transitive closure of similarity, and similarity is not transitive, so the
pipeline was taking a closure of an already-closed relation. Measured on the
giant chain cluster (2,474 chains): 3,059,101 pairs were treated as similar,
715,169 actually clear ``>= 0.24`` — **76.6% were linked only by a path**.
``1bj1`` (VEGF + antibody) and ``1e6e`` (adrenodoxin reductase + adrenodoxin)
landed in one cluster with no similar chain between them, joined by three hops.
Historical folds were therefore over-conservative, not leaky.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import pandas as pd

from ppisplit.types import ChainKey, ComplexInfo


class UnionFind:
    """Disjoint-set / union-find for computing connected components."""

    def __init__(self, elements: Iterable[str]) -> None:
        self.parent: dict[str, str] = {e: e for e in elements}
        self.rank: dict[str, int] = {e: 0 for e in self.parent}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def components(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for e in self.parent:
            groups[self.find(e)].append(e)
        return dict(groups)


def chain_clusters_from_edges(
    edges: pd.DataFrame,
    all_chains: Iterable[ChainKey],
) -> dict[ChainKey, int]:
    """Connected components on a chain similarity graph.

    ``edges`` must have ``query`` and ``target`` columns. Chains in
    ``all_chains`` with no edges become singleton clusters.

    Returns ``{chain_key: cluster_id}`` with cluster_ids assigned by
    sorted-rep order for determinism.
    """
    uf = UnionFind(all_chains)
    if not edges.empty:
        for q, t in edges[["query", "target"]].itertuples(index=False, name=None):
            if q in uf.parent and t in uf.parent:
                uf.union(q, t)

    components = uf.components()
    chain_clusters: dict[ChainKey, int] = {}
    for cluster_id, (_, members) in enumerate(sorted(components.items())):
        for ch in members:
            chain_clusters[ch] = cluster_id
    return chain_clusters


_VALID_SIMILARITY_MODES = ("both", "receptor-only", "ligand-only", "either")


def similarity_map(edges: pd.DataFrame) -> dict[ChainKey, set[ChainKey]]:
    """``{chain: {chains directly similar to it}}`` from an edge table.

    Symmetrized, because the relation is symmetric even when mmseqs only
    reported one direction. This is the *direct* relation — no closure.
    """
    sim: dict[ChainKey, set[ChainKey]] = defaultdict(set)
    if edges.empty:
        return sim
    for q, t in edges[["query", "target"]].itertuples(index=False, name=None):
        if q == t:
            continue
        sim[q].add(t)
        sim[t].add(q)
    return sim


def _side_match(
    side_a: Iterable[ChainKey],
    side_b: Iterable[ChainKey],
    sim: dict[ChainKey, set[ChainKey]],
) -> bool:
    """True if any chain of ``side_a`` is directly similar to one of ``side_b``."""
    side_b = set(side_b)
    return any(a in sim and sim[a] & side_b for a in side_a)


def complexes_are_similar(
    a: ComplexInfo,
    b: ComplexInfo,
    sim: dict[ChainKey, set[ChainKey]],
    similarity_mode: str = "both",
) -> bool:
    """Direct both-mode (or single-side) similarity test between two complexes.

    'both' requires a receptor~receptor **and** a ligand~ligand match, or the
    swapped-annotation equivalent. It is the same rule
    :func:`ppisplit.leaks.detect_leaks` applies, so fold boundaries and
    test-leak verdicts are governed by one relation.
    """
    rec_a, lig_a = a.receptor_keys(), a.ligand_keys()
    rec_b, lig_b = b.receptor_keys(), b.ligand_keys()

    if similarity_mode == "receptor-only":
        return _side_match(rec_a, rec_b, sim)
    if similarity_mode == "ligand-only":
        return _side_match(lig_a, lig_b, sim)
    if similarity_mode == "either":
        return _side_match(a.chain_keys(), b.chain_keys(), sim)
    return (
        (_side_match(rec_a, rec_b, sim) and _side_match(lig_a, lig_b, sim))
        or (_side_match(rec_a, lig_b, sim) and _side_match(lig_a, rec_b, sim))
    )


def complex_clusters_from_edges(
    complexes: Iterable[ComplexInfo],
    edges: pd.DataFrame,
    similarity_mode: str = "both",
    known_chains: set[ChainKey] | None = None,
) -> dict[str, int]:
    """Connected components over complexes joined by *direct* chain edges.

    ``edges`` is the post-filtered hit table (``query``/``target`` chain keys).
    Two complexes are adjacent iff they satisfy
    :func:`complexes_are_similar` — no chain-cluster intermediate, so the only
    transitive closure is the single, intended one at complex level.

    Candidate pairs come from an inverted ``chain -> complexes`` index over the
    similarity neighbourhoods, so the cost is proportional to the edge count,
    not to N². Complexes whose chains are all absent from ``known_chains``
    (when given) are skipped, matching the old behaviour for UIDs missing from
    the search universe.

    Returns ``{uid: complex_cluster_id}``.
    """
    if similarity_mode not in _VALID_SIMILARITY_MODES:
        raise ValueError(
            f"similarity_mode must be one of {_VALID_SIMILARITY_MODES}, got {similarity_mode!r}"
        )

    kept: list[ComplexInfo] = []
    for c in complexes:
        if known_chains is not None and not (set(c.chain_keys()) & known_chains):
            continue
        kept.append(c)

    by_uid = {c.uid: c for c in kept}
    sim = similarity_map(edges)

    chain_to_uids: dict[ChainKey, set[str]] = defaultdict(set)
    for c in kept:
        for key in c.chain_keys():
            chain_to_uids[key].add(c.uid)

    uf = UnionFind(sorted(by_uid))
    for c in kept:
        neighbours: set[str] = set()
        for key in c.chain_keys():
            for similar_chain in sim.get(key, ()):
                neighbours |= chain_to_uids.get(similar_chain, set())
        for other_uid in neighbours:
            if other_uid == c.uid:
                continue
            if complexes_are_similar(c, by_uid[other_uid], sim, similarity_mode):
                uf.union(c.uid, other_uid)

    components = uf.components()
    uid_to_cluster: dict[str, int] = {}
    for cluster_id, (_, members) in enumerate(sorted(components.items())):
        for uid in members:
            uid_to_cluster[uid] = cluster_id
    return uid_to_cluster


def group_clusters(uid_to_cluster: dict[str, int]) -> dict[int, list[str]]:
    """Invert ``{uid: cluster_id}`` to ``{cluster_id: [uids]}``."""
    grouped: dict[int, list[str]] = defaultdict(list)
    for uid, cid in uid_to_cluster.items():
        grouped[cid].append(uid)
    return dict(grouped)


def exclude_test_clusters(
    complex_clusters: dict[int, list[str]],
    test_uids: set[str],
) -> tuple[dict[int, list[str]], dict[int, list[str]], list[str]]:
    """Separate clusters into trainval and test groups.

    Any cluster containing a test UID is removed entirely from trainval.
    Returns ``(trainval_clusters, test_clusters, collateral_uids)`` where
    ``collateral_uids`` are train-set members caught in test clusters.
    """
    trainval: dict[int, list[str]] = {}
    test: dict[int, list[str]] = {}
    collateral: list[str] = []

    for cid, uids in complex_clusters.items():
        if any(u in test_uids for u in uids):
            test[cid] = uids
            collateral.extend(u for u in uids if u not in test_uids)
        else:
            trainval[cid] = uids

    return trainval, test, collateral
