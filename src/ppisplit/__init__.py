"""ppisplit: sequence-aware CV splitting and leak detection for PPI datasets.

Pipeline (architecturally decoupled):
    1. Extract chain sequences from PDB files (resolved residues by default).
    2. Run mmseqs2 easy-search at fixed *recall* settings — one (query, target)
       database pair per question, never pooled.
    3. Apply an EdgePredicate to the hit table to derive chain-level similarity edges.
    4. Complex-level union-find over the *direct* chain edges -> complex clusters.
    5. Exclude test-collateral clusters; assign remaining clusters to CV folds.
    6. Same hit table + EdgePredicate also drives leak detection.

Chain clusters (step 4's removed predecessor) are a diagnostic only: cluster
co-membership is the transitive closure of similarity and must never be used as
a similarity proxy. See :mod:`ppisplit.graph`.

Public API exposes each component independently so callers can swap
implementations without rewriting the orchestration.
"""

from ppisplit.folds import assign_kfold, assign_montecarlo
from ppisplit.graph import (
    UnionFind,
    chain_clusters_from_edges,
    complex_clusters_from_edges,
    complexes_are_similar,
    exclude_test_clusters,
    similarity_map,
)
from ppisplit.io import (
    DEFAULT_SEQUENCE_SOURCE,
    SEQUENCE_SOURCES,
    extract_sequences_from_pdb,
    write_fasta,
)
from ppisplit.leaks import LeakReport, detect_leaks
from ppisplit.predicates import (
    EdgePredicate,
    HyperbolaPredicate,
    RectanglePredicate,
    UnionOfPredicates,
)
from ppisplit.search import (
    SEARCH_SETTINGS,
    run_mmseqs_search_allvall,
    run_mmseqs_search_pair,
)
from ppisplit.types import ChainKey, ComplexInfo, Hit

__all__ = [
    "ChainKey",
    "ComplexInfo",
    "DEFAULT_SEQUENCE_SOURCE",
    "EdgePredicate",
    "Hit",
    "HyperbolaPredicate",
    "LeakReport",
    "RectanglePredicate",
    "SEARCH_SETTINGS",
    "SEQUENCE_SOURCES",
    "UnionFind",
    "UnionOfPredicates",
    "assign_kfold",
    "assign_montecarlo",
    "chain_clusters_from_edges",
    "complex_clusters_from_edges",
    "complexes_are_similar",
    "detect_leaks",
    "exclude_test_clusters",
    "extract_sequences_from_pdb",
    "run_mmseqs_search_allvall",
    "run_mmseqs_search_pair",
    "similarity_map",
    "write_fasta",
]
