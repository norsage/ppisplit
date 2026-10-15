import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from ppisplit.graph import (
    UnionFind,
    chain_clusters_from_edges,
    complex_clusters_from_edges,
    complexes_are_similar,
    exclude_test_clusters,
    group_clusters,
    numbered_components,
    similarity_map,
)
from ppisplit.types import ComplexInfo


def edge_frame(pairs):
    return pd.DataFrame(pairs, columns=["query", "target"])


def test_union_find_basic_components():
    uf = UnionFind(["a", "b", "c", "d"])
    uf.union("a", "b")
    uf.union("c", "d")
    components = uf.components()
    sizes = sorted(len(v) for v in components.values())
    assert sizes == [2, 2]


def test_chain_clusters_with_singletons():
    edges = pd.DataFrame({
        "query":  ["x_A", "x_B", "y_A"],
        "target": ["y_A", "y_B", "z_A"],
    })
    all_chains = ["x_A", "x_B", "y_A", "y_B", "z_A", "lonely"]
    cc = chain_clusters_from_edges(edges, all_chains)
    # x_A-y_A-z_A connected; x_B-y_B connected; lonely is its own cluster.
    assert cc["x_A"] == cc["y_A"] == cc["z_A"]
    assert cc["x_B"] == cc["y_B"]
    assert cc["lonely"] not in {cc["x_A"], cc["x_B"]}
    # Three distinct clusters total.
    assert len(set(cc.values())) == 3


def test_chain_clusters_empty_edges():
    cc = chain_clusters_from_edges(
        pd.DataFrame(columns=["query", "target"]), ["a", "b", "c"]
    )
    # Every chain is its own singleton.
    assert len(set(cc.values())) == 3


def test_similarity_map_is_symmetric_and_direct():
    sim = similarity_map(edge_frame([("a", "b"), ("b", "c")]))
    assert sim["a"] == {"b"}
    assert sim["b"] == {"a", "c"}
    # No closure: a and c are not similar just because both touch b.
    assert "c" not in sim["a"]


def test_complex_clusters_both_mode_same_role():
    complexes = [
        ComplexInfo("c1", receptor_chains=("A",), ligand_chains=("B",)),
        ComplexInfo("c2", receptor_chains=("A",), ligand_chains=("B",)),
        ComplexInfo("c3", receptor_chains=("X",), ligand_chains=("Y",)),
    ]
    edges = edge_frame([("c1_A", "c2_A"), ("c1_B", "c2_B")])
    uid_to_cluster = complex_clusters_from_edges(complexes, edges, "both")
    assert uid_to_cluster["c1"] == uid_to_cluster["c2"]
    assert uid_to_cluster["c3"] != uid_to_cluster["c1"]


def test_complex_clusters_both_mode_swapped_roles():
    # c1: receptor=R, ligand=L.  c2: receptor=L, ligand=R (annotation swap).
    complexes = [
        ComplexInfo("c1", receptor_chains=("R",), ligand_chains=("L",)),
        ComplexInfo("c2", receptor_chains=("L",), ligand_chains=("R",)),
    ]
    edges = edge_frame([("c1_R", "c2_R"), ("c1_L", "c2_L")])
    uid_to_cluster = complex_clusters_from_edges(complexes, edges, "both")
    assert uid_to_cluster["c1"] == uid_to_cluster["c2"]


def test_complex_clusters_both_mode_partial_match_doesnt_merge():
    complexes = [
        ComplexInfo("c1", receptor_chains=("A",), ligand_chains=("B",)),
        ComplexInfo("c2", receptor_chains=("A",), ligand_chains=("Z",)),
    ]
    edges = edge_frame([("c1_A", "c2_A")])  # receptors only
    uid_to_cluster = complex_clusters_from_edges(complexes, edges, "both")
    assert uid_to_cluster["c1"] != uid_to_cluster["c2"]


def test_complex_clusters_receptor_only_mode():
    complexes = [
        ComplexInfo("c1", receptor_chains=("A",), ligand_chains=("B",)),
        ComplexInfo("c2", receptor_chains=("A",), ligand_chains=("Z",)),
    ]
    edges = edge_frame([("c1_A", "c2_A")])
    uid_to_cluster = complex_clusters_from_edges(complexes, edges, "receptor-only")
    assert uid_to_cluster["c1"] == uid_to_cluster["c2"]


def test_complex_clusters_rejects_unknown_mode():
    with pytest.raises(ValueError):
        complex_clusters_from_edges([], edge_frame([]), "nonsense")


def test_complex_clusters_ignore_edges_to_chains_outside_the_annotation():
    complexes = [ComplexInfo("c1", ("A",), ("B",)), ComplexInfo("c2", ("A",), ("B",))]
    edges = edge_frame([("c1_A", "other_A"), ("c1_B", "other_B")])
    uid_to_cluster = complex_clusters_from_edges(complexes, edges, "both")
    assert uid_to_cluster["c1"] != uid_to_cluster["c2"]


def test_complexes_with_no_known_chains_are_skipped():
    complexes = [ComplexInfo("c1", ("A",), ("B",)), ComplexInfo("ghost", ("A",), ("B",))]
    uid_to_cluster = complex_clusters_from_edges(
        complexes, edge_frame([]), "both", known_chains={"c1_A", "c1_B"}
    )
    assert set(uid_to_cluster) == {"c1"}


# ---------------------------------------------------------------------------
# The chain-cluster defect this module was rewritten to remove.
# ---------------------------------------------------------------------------


def test_chain_cluster_co_membership_is_never_used_as_similarity():
    """Transitively-linked chains must not, by themselves, merge complexes.

    The real failure: `1bj1` (VEGF + antibody) and `1e6e` (adrenodoxin
    reductase + adrenodoxin) shared a chain cluster with no similar chain
    between them, joined by a three-hop path. Here c1-m1-m2-c2 on both sides
    puts every chain in one chain cluster per side, but no chain of c1 is
    directly similar to any chain of c2.
    """
    complexes = [
        ComplexInfo("c1", receptor_chains=("A",), ligand_chains=("B",)),
        ComplexInfo("c2", receptor_chains=("A",), ligand_chains=("B",)),
        ComplexInfo("m1", receptor_chains=("A",), ligand_chains=("B",)),
        ComplexInfo("m2", receptor_chains=("A",), ligand_chains=("B",)),
    ]
    edges = edge_frame([
        ("c1_A", "m1_A"), ("m1_A", "m2_A"), ("m2_A", "c2_A"),
        ("c1_B", "m1_B"), ("m1_B", "m2_B"), ("m2_B", "c2_B"),
    ])

    # Every chain of c1 and c2 sits in one chain cluster per side...
    all_chains = [k for c in complexes for k in c.chain_keys()]
    chain_clusters = chain_clusters_from_edges(edges, all_chains)
    assert chain_clusters["c1_A"] == chain_clusters["c2_A"]
    assert chain_clusters["c1_B"] == chain_clusters["c2_B"]

    # ...yet c1 and c2 are not directly similar, so nothing may treat them as
    # adjacent on that basis.
    sim = similarity_map(edges)
    assert not complexes_are_similar(complexes[0], complexes[1], sim, "both")

    # Drop the intermediates and no merge happens at all — under the old
    # chain-cluster proxy the two would still have been merged.
    isolated = complex_clusters_from_edges(complexes[:2], edges, "both")
    assert isolated["c1"] != isolated["c2"]

    # With the intermediates present, complex-level connected components do
    # merge all four — via the real chain of direct adjacencies c1-m1-m2-c2.
    # That is the intended closure, taken exactly once, at complex level.
    uid_to_cluster = complex_clusters_from_edges(complexes, edges, "both")
    assert len(set(uid_to_cluster.values())) == 1


def test_exclude_test_clusters_removes_collateral():
    grouped = {0: ["t1", "x1", "x2"], 1: ["x3"], 2: ["t2"]}
    test_uids = {"t1", "t2"}
    trainval, test, collateral = exclude_test_clusters(grouped, test_uids)
    assert set(trainval.keys()) == {1}
    assert set(test.keys()) == {0, 2}
    assert sorted(collateral) == ["x1", "x2"]


def test_group_clusters_inversion():
    uid_to_cluster = {"a": 0, "b": 0, "c": 1}
    grouped = group_clusters(uid_to_cluster)
    assert grouped == {0: ["a", "b"], 1: ["c"]} or grouped == {0: ["b", "a"], 1: ["c"]}


def test_numbered_components_do_not_depend_on_union_order():
    forward, backward = UnionFind(["a", "b", "c", "d", "e"]), UnionFind(["a", "b", "c", "d", "e"])
    for x, y in [("c", "a"), ("d", "b"), ("e", "d")]:
        forward.union(x, y)
    for x, y in [("d", "e"), ("b", "d"), ("a", "c")]:
        backward.union(x, y)
    want = [["a", "c"], ["b", "d", "e"]]
    assert [sorted(m) for m in numbered_components(forward)] == want
    assert [sorted(m) for m in numbered_components(backward)] == want


CLUSTERS_SCRIPT = """
import json
import random
import pandas as pd
from ppisplit.graph import complex_clusters_from_edges
from ppisplit.types import ComplexInfo

# a sparse random graph: some complexes join two trees of equal rank, and which
# root survives follows the iteration order of a set of uids
rng = random.Random(0)
uids = [f"u{i:03d}" for i in range(200)]
complexes = [ComplexInfo(u, receptor_chains=("A",), ligand_chains=("B",)) for u in uids]
pairs = []
for _ in range(150):
    a, b = rng.sample(uids, 2)
    pairs += [(f"{a}_A", f"{b}_A"), (f"{a}_B", f"{b}_B")]
edges = pd.DataFrame(pairs, columns=["query", "target"])
print(json.dumps(complex_clusters_from_edges(complexes, edges, "both"), sort_keys=True))
"""


def test_complex_cluster_ids_do_not_depend_on_hash_seed():
    outputs = set()
    for seed in ("1", "2", "3", "4"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        run = subprocess.run(
            [sys.executable, "-c", CLUSTERS_SCRIPT], env=env, capture_output=True, text=True, check=True
        )
        outputs.add(run.stdout)
    assert len(outputs) == 1
    clusters = json.loads(outputs.pop())
    smallest = [min(u for u, c in clusters.items() if c == k) for k in range(max(clusters.values()) + 1)]
    assert smallest == sorted(smallest)
