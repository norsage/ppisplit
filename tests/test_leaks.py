import pandas as pd

from ppisplit.leaks import detect_leaks
from ppisplit.predicates import HyperbolaPredicate, RectanglePredicate


def make_chain_role(complexes_def):
    """complexes_def: list of (uid, [receptor_chains], [ligand_chains])."""
    role = {}
    for uid, recs, ligs in complexes_def:
        for ch in recs:
            role[f"{uid}_{ch}"] = "receptor"
        for ch in ligs:
            role[f"{uid}_{ch}"] = "ligand"
    return role


def hit_row(query, target, pident=0.9, qcov=0.9, tcov=0.9, alnlen=200, qlen=220, tlen=220, evalue=1e-10):
    return {
        "query": query, "target": target,
        "pident": pident, "alnlen": alnlen,
        "qlen": qlen, "tlen": tlen,
        "qcov": qcov, "tcov": tcov,
        "evalue": evalue, "fident": pident, "bits": 100,
    }


def test_same_role_both_mode_requires_receptor_and_ligand_match():
    # test=t1 (R=A, L=B), train=n1 (R=A, L=B). Both partners match.
    chain_role = make_chain_role([("t1", ["A"], ["B"]), ("n1", ["A"], ["B"])])
    hits = pd.DataFrame([
        hit_row("t1_A", "n1_A"),
        hit_row("t1_B", "n1_B"),
    ])
    pred = RectanglePredicate(0.3, 0.8, cov_mode=1)
    report = detect_leaks(hits, pred, chain_role, {"t1"}, {"n1"}, "both")
    assert list(report.leaking_pairs.itertuples(index=False, name=None)) == [("t1", "n1")]


def test_same_role_both_mode_only_receptor_match_is_not_a_leak():
    # Receptor matches but ligand doesn't.
    chain_role = make_chain_role([("t1", ["A"], ["B"]), ("n1", ["A"], ["Z"])])
    hits = pd.DataFrame([hit_row("t1_A", "n1_A")])
    pred = RectanglePredicate(0.3, 0.8, cov_mode=1)
    report = detect_leaks(hits, pred, chain_role, {"t1"}, {"n1"}, "both")
    assert report.leaking_pairs.empty


def test_swapped_role_both_mode_detects_cross_leak():
    # test=t1 (R=A, L=B), train=n1 (R=B, L=A) -- annotation roles swapped.
    chain_role = make_chain_role([("t1", ["A"], ["B"]), ("n1", ["B"], ["A"])])
    hits = pd.DataFrame([
        hit_row("t1_A", "n1_A"),  # test receptor ~ train ligand (cross)
        hit_row("t1_B", "n1_B"),  # test ligand   ~ train receptor (cross)
    ])
    pred = RectanglePredicate(0.3, 0.8, cov_mode=1)
    report = detect_leaks(hits, pred, chain_role, {"t1"}, {"n1"}, "both")
    assert list(report.leaking_pairs.itertuples(index=False, name=None)) == [("t1", "n1")]


def test_swapped_role_one_direction_only_is_not_a_leak():
    # Only test-receptor ~ train-ligand; the reverse direction has no hit.
    chain_role = make_chain_role([("t1", ["A"], ["B"]), ("n1", ["B"], ["X"])])
    hits = pd.DataFrame([hit_row("t1_A", "n1_X")])  # cross-role, single direction only
    pred = RectanglePredicate(0.3, 0.8, cov_mode=1)
    report = detect_leaks(hits, pred, chain_role, {"t1"}, {"n1"}, "both")
    assert report.leaking_pairs.empty


def test_predicate_filters_below_threshold_hits():
    chain_role = make_chain_role([("t1", ["A"], ["B"]), ("n1", ["A"], ["B"])])
    hits = pd.DataFrame([
        hit_row("t1_A", "n1_A", pident=0.25, tcov=0.9),  # below 0.30 identity
        hit_row("t1_B", "n1_B", pident=0.95, tcov=0.5),  # below 0.80 coverage
    ])
    pred = RectanglePredicate(0.30, 0.80, cov_mode=1)
    report = detect_leaks(hits, pred, chain_role, {"t1"}, {"n1"}, "both")
    assert report.leaking_pairs.empty


def test_either_mode_any_same_role_match_is_a_leak():
    chain_role = make_chain_role([("t1", ["A"], ["B"]), ("n1", ["A"], ["Z"])])
    hits = pd.DataFrame([hit_row("t1_A", "n1_A")])
    pred = RectanglePredicate(0.3, 0.8, cov_mode=1)
    report = detect_leaks(hits, pred, chain_role, {"t1"}, {"n1"}, "either")
    assert list(report.leaking_pairs.itertuples(index=False, name=None)) == [("t1", "n1")]


def test_evidence_records_underlying_hits():
    chain_role = make_chain_role([("t1", ["A"], ["B"]), ("n1", ["A"], ["B"])])
    hits = pd.DataFrame([
        hit_row("t1_A", "n1_A"),
        hit_row("t1_B", "n1_B"),
    ])
    pred = RectanglePredicate(0.3, 0.8, cov_mode=1)
    report = detect_leaks(hits, pred, chain_role, {"t1"}, {"n1"}, "both")
    assert len(report.evidence) == 2
    assert set(report.evidence["query_role"]) == {"receptor", "ligand"}


def test_direct_uid_overlap_is_a_leak_even_with_no_hits():
    # 5vnw-style case: same UID in both test and train, no other sequence-similar
    # neighbours. Predicate path returns nothing (chain-key self-hits filtered),
    # but the direct overlap should still flag the pair.
    chain_role = make_chain_role([("5vnw", ["A"], ["D"]), ("other", ["X"], ["Y"])])
    hits = pd.DataFrame(columns=[
        "query", "target", "fident", "pident", "alnlen", "qlen", "tlen",
        "qcov", "tcov", "evalue", "bits",
    ])
    pred = RectanglePredicate(0.3, 0.8, cov_mode=1)
    report = detect_leaks(hits, pred, chain_role, {"5vnw"}, {"5vnw", "other"}, "both")
    assert list(report.leaking_pairs.itertuples(index=False, name=None)) == [("5vnw", "5vnw")]
    assert report.leaking_test == {"5vnw"}
    assert report.leaking_train == {"5vnw"}


def test_direct_uid_overlap_unions_with_predicate_leaks():
    # Both a direct duplicate AND a sequence-similar leak from a different train UID.
    chain_role = make_chain_role([
        ("dupe", ["A"], ["B"]),
        ("similar", ["A"], ["B"]),
    ])
    hits = pd.DataFrame([
        hit_row("dupe_A", "similar_A"),
        hit_row("dupe_B", "similar_B"),
    ])
    pred = RectanglePredicate(0.3, 0.8, cov_mode=1)
    report = detect_leaks(hits, pred, chain_role, {"dupe"}, {"dupe", "similar"}, "both")
    pairs = set(report.leaking_pairs.itertuples(index=False, name=None))
    assert pairs == {("dupe", "dupe"), ("dupe", "similar")}


def test_direct_uid_overlap_appears_in_evidence():
    # Regression: exact PDB-ID duplicates have no mmseqs hit, but must still be
    # recorded in evidence (tagged leak_type='direct-uid') so the output CSV is
    # a faithful record and downstream consumers don't undercount.
    chain_role = make_chain_role([("5vnw", ["A"], ["D"]), ("other", ["X"], ["Y"])])
    hits = pd.DataFrame(columns=[
        "query", "target", "fident", "pident", "alnlen", "qlen", "tlen",
        "qcov", "tcov", "evalue", "bits",
    ])
    pred = RectanglePredicate(0.3, 0.8, cov_mode=1)
    report = detect_leaks(hits, pred, chain_role, {"5vnw"}, {"5vnw", "other"}, "both")
    assert "leak_type" in report.evidence.columns
    direct = report.evidence[report.evidence["leak_type"] == "direct-uid"]
    assert set(zip(direct["query_uid"], direct["target_uid"])) == {("5vnw", "5vnw")}


def test_evidence_tags_sequence_and_direct_leaks_distinctly():
    chain_role = make_chain_role([
        ("dupe", ["A"], ["B"]),
        ("similar", ["A"], ["B"]),
    ])
    hits = pd.DataFrame([
        hit_row("dupe_A", "similar_A"),
        hit_row("dupe_B", "similar_B"),
    ])
    pred = RectanglePredicate(0.3, 0.8, cov_mode=1)
    report = detect_leaks(hits, pred, chain_role, {"dupe"}, {"dupe", "similar"}, "both")
    types = set(report.evidence["leak_type"])
    assert types == {"sequence", "direct-uid"}
    direct = report.evidence[report.evidence["leak_type"] == "direct-uid"]
    assert set(zip(direct["query_uid"], direct["target_uid"])) == {("dupe", "dupe")}


def test_train_to_test_direction_is_ignored():
    # detect_leaks should only consider (test query -> train target) edges.
    chain_role = make_chain_role([("t1", ["A"], ["B"]), ("n1", ["A"], ["B"])])
    hits = pd.DataFrame([
        hit_row("n1_A", "t1_A"),  # train query -> test target -- wrong direction
        hit_row("n1_B", "t1_B"),
    ])
    pred = RectanglePredicate(0.3, 0.8, cov_mode=1)
    report = detect_leaks(hits, pred, chain_role, {"t1"}, {"n1"}, "both")
    assert report.leaking_pairs.empty


def test_underscored_uids_are_matched_in_every_mode():
    # Regression for 0004.k7l / 0003.5un: uids with "_" (`1jck_a_b`) were cut at
    # the first "_", so no edge matched a uid and every mode reported nothing.
    chain_role = make_chain_role(
        [("2aq3_a_b", ["A"], ["B"]), ("1jck_a_b", ["A"], ["B"])]
    )
    hits = pd.DataFrame([
        hit_row("2aq3_a_b_A", "1jck_a_b_A", pident=0.89, qcov=1.0, tcov=1.0),
        hit_row("2aq3_a_b_B", "1jck_a_b_B", pident=0.95, qcov=1.0, tcov=1.0),
    ])
    pred = HyperbolaPredicate(0.24)
    for mode in ("both", "either", "receptor-only", "ligand-only"):
        report = detect_leaks(hits, pred, chain_role, {"2aq3_a_b"}, {"1jck_a_b"}, mode)
        assert list(report.leaking_pairs.itertuples(index=False, name=None)) == [
            ("2aq3_a_b", "1jck_a_b")
        ], mode
        assert set(report.evidence["query_chain"]) == {"A", "B"}


def test_self_check_reports_leak():
    chain_role = make_chain_role([("1jck_a_b", ["A"], ["B"])])
    hits = pd.DataFrame([
        hit_row("1jck_a_b_A", "1jck_a_b_A", pident=1.0, qcov=1.0, tcov=1.0),
        hit_row("1jck_a_b_B", "1jck_a_b_B", pident=1.0, qcov=1.0, tcov=1.0),
    ])
    report = detect_leaks(hits, HyperbolaPredicate(0.24), chain_role, {"1jck_a_b"}, {"1jck_a_b"})
    assert not report.leaking_pairs.empty
