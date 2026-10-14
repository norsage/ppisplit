"""Leak detection: which (test_uid, train_uid) pairs are sequence-similar.

Canonical rule (see the README): a test complex leaks when
**both** of its partners match the same train complex under the edge predicate —
``fident * max(qcov, tcov) >= 0.24``. There is no separate antigen-identity
threshold; one matching partner is by design not a leak. The predicate object is
the same one used to build CV clusters, so train-val and train-test leakage are
governed consistently.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ppisplit.predicates import EdgePredicate
from ppisplit.types import ChainKey


_VALID_SIMILARITY_MODES = ("both", "receptor-only", "ligand-only", "either")

# Columns carried in the ``evidence`` DataFrame in addition to the raw mmseqs
# hit columns. ``leak_type`` distinguishes sequence-similarity leaks ("sequence")
# from exact PDB-ID duplicates ("direct-uid").
_LEAK_TYPE_COL = "leak_type"


def _direct_evidence(direct_overlap: set[str], columns: list[str]) -> pd.DataFrame:
    """Synthesise one evidence row per exact test/train PDB-ID duplicate.

    These pairs have no mmseqs hit (self-hits are removed at the chain-key
    level), so without this the output evidence silently omits the strongest
    possible leaks. Rows are marked ``leak_type='direct-uid'`` with identity
    100% / full coverage to reflect that the structure is literally shared.
    """
    if not direct_overlap:
        return pd.DataFrame(columns=columns)
    rows = [
        {
            "query": uid,
            "target": uid,
            "query_uid": uid,
            "target_uid": uid,
            "query_chain": pd.NA,
            "target_chain": pd.NA,
            "query_role": pd.NA,
            "target_role": pd.NA,
            "pident": 100.0,
            "fident": 1.0,
            "alnlen": pd.NA,
            "qlen": pd.NA,
            "tlen": pd.NA,
            "qcov": 1.0,
            "tcov": 1.0,
            "evalue": 0.0,
            _LEAK_TYPE_COL: "direct-uid",
        }
        for uid in sorted(direct_overlap)
    ]
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class LeakReport:
    """Output of :func:`detect_leaks`.

    Attributes:
        leaking_pairs: DataFrame[test_uid, train_uid] of distinct leaks.
        leaking_test:  set of test UIDs involved in any leaking pair.
        leaking_train: set of train UIDs involved in any leaking pair.
        evidence:      DataFrame of all individual hits underlying the leaks,
                       with role annotations and the original mmseqs columns.
    """

    leaking_pairs: pd.DataFrame
    leaking_test: set[str]
    leaking_train: set[str]
    evidence: pd.DataFrame


def detect_leaks(
    hits: pd.DataFrame,
    predicate: EdgePredicate,
    chain_role: dict[ChainKey, str],
    test_uids: set[str],
    train_uids: set[str],
    similarity_mode: str = "both",
) -> LeakReport:
    """Identify (test_uid, train_uid) pairs that constitute leaks.

    A pair is a leak under ``similarity_mode='both'`` if either:
      - same-role: receptor~receptor AND ligand~ligand sequence matches, OR
      - swapped:   test_receptor~train_ligand AND test_ligand~train_receptor.

    Other modes:
      - 'receptor-only': any receptor-receptor match
      - 'ligand-only':   any ligand-ligand match
      - 'either':        any same-role match
    """
    if similarity_mode not in _VALID_SIMILARITY_MODES:
        raise ValueError(
            f"similarity_mode must be one of {_VALID_SIMILARITY_MODES}, got {similarity_mode!r}"
        )

    # Direct UID overlap is the strongest possible leak — same complex appears in
    # both test and train. Pairwise mmseqs hits get filtered as self-matches at
    # the chain-key level, so the predicate-based path misses these without an
    # explicit check.
    direct_overlap = test_uids & train_uids

    evidence_cols = [*hits.columns.tolist(), _LEAK_TYPE_COL]

    def _direct_only_report() -> LeakReport:
        if not direct_overlap:
            return LeakReport(
                leaking_pairs=pd.DataFrame(columns=["test_uid", "train_uid"]),
                leaking_test=set(),
                leaking_train=set(),
                evidence=pd.DataFrame(columns=evidence_cols),
            )
        direct_df = pd.DataFrame(
            {"test_uid": sorted(direct_overlap), "train_uid": sorted(direct_overlap)}
        )
        return LeakReport(
            leaking_pairs=direct_df,
            leaking_test=set(direct_overlap),
            leaking_train=set(direct_overlap),
            evidence=_direct_evidence(direct_overlap, evidence_cols),
        )

    edges = hits[predicate(hits)].copy()
    if edges.empty:
        return _direct_only_report()

    edges["query_uid"] = edges["query"].str.split("_").str[0]
    edges["query_chain"] = edges["query"].str.split("_").str[1]
    edges["target_uid"] = edges["target"].str.split("_").str[0]
    edges["target_chain"] = edges["target"].str.split("_").str[1]
    edges["query_role"] = edges["query"].map(chain_role)
    edges["target_role"] = edges["target"].map(chain_role)
    edges = edges.dropna(subset=["query_role", "target_role"])

    # Restrict to (test query, train target). This is a choice of *which row to
    # report*, not a filter that drops edges: both search paths cover the pair
    # in both directions (run_mmseqs_search_pair merges the two runs,
    # run_mmseqs_search_allvall symmetrizes), and the canonical predicate scores
    # max(qcov, tcov), which is orientation invariant. mmseqs does sometimes
    # report genuinely different alignments for the two directions (~15% of
    # pairs, 0.3% of them straddling x=0.24), so the two orientations are not
    # perfectly interchangeable — but taking the higher-scoring orientation
    # instead changes no verdict on the current sets.
    edges = edges[
        edges["query_uid"].isin(test_uids) & edges["target_uid"].isin(train_uids)
    ].reset_index(drop=True)
    if edges.empty:
        return _direct_only_report()

    same_role = edges[edges["query_role"] == edges["target_role"]]
    cross_role = edges[edges["query_role"] != edges["target_role"]]

    same_pairs = (
        same_role.groupby(["query_uid", "target_uid"])["query_role"]
        .apply(set).reset_index().rename(columns={"query_role": "matching_roles"})
        if not same_role.empty
        else pd.DataFrame(columns=["query_uid", "target_uid", "matching_roles"])
    )
    cross_pairs = (
        cross_role.groupby(["query_uid", "target_uid"])["query_role"]
        .apply(set).reset_index().rename(columns={"query_role": "cross_roles"})
        if not cross_role.empty
        else pd.DataFrame(columns=["query_uid", "target_uid", "cross_roles"])
    )

    if similarity_mode == "both":
        same_leaks = (
            same_pairs[same_pairs["matching_roles"] >= {"receptor", "ligand"}]
            if not same_pairs.empty else same_pairs
        )
        cross_leaks = (
            cross_pairs[cross_pairs["cross_roles"] >= {"receptor", "ligand"}]
            if not cross_pairs.empty else cross_pairs
        )
        leaking_pairs = pd.concat(
            [same_leaks[["query_uid", "target_uid"]], cross_leaks[["query_uid", "target_uid"]]],
            ignore_index=True,
        ).drop_duplicates()
    elif similarity_mode == "receptor-only":
        leaking_pairs = (
            same_pairs[same_pairs["matching_roles"].apply(lambda s: "receptor" in s)]
            [["query_uid", "target_uid"]]
        )
    elif similarity_mode == "ligand-only":
        leaking_pairs = (
            same_pairs[same_pairs["matching_roles"].apply(lambda s: "ligand" in s)]
            [["query_uid", "target_uid"]]
        )
    else:  # 'either'
        leaking_pairs = same_pairs[["query_uid", "target_uid"]] if not same_pairs.empty else same_pairs

    leaking_pairs = leaking_pairs.rename(
        columns={"query_uid": "test_uid", "target_uid": "train_uid"}
    ).reset_index(drop=True)

    pair_keys = set(zip(leaking_pairs["test_uid"], leaking_pairs["train_uid"]))
    evidence = edges[
        edges.apply(lambda r: (r["query_uid"], r["target_uid"]) in pair_keys, axis=1)
    ].reset_index(drop=True)
    evidence[_LEAK_TYPE_COL] = "sequence"

    if direct_overlap:
        direct_df = pd.DataFrame(
            {"test_uid": sorted(direct_overlap), "train_uid": sorted(direct_overlap)}
        )
        leaking_pairs = (
            pd.concat([leaking_pairs, direct_df], ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )
        # Exact PDB-ID duplicates have no mmseqs hit (self-hits removed), so add
        # explicit evidence rows — otherwise the strongest leaks are omitted.
        evidence = pd.concat(
            [evidence, _direct_evidence(direct_overlap, evidence_cols)],
            ignore_index=True,
        )

    return LeakReport(
        leaking_pairs=leaking_pairs,
        leaking_test=set(leaking_pairs["test_uid"]),
        leaking_train=set(leaking_pairs["train_uid"]),
        evidence=evidence,
    )
