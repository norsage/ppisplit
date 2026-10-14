import pandas as pd

from ppisplit.predicates import (
    COV_MODE_SYMMETRIC,
    HyperbolaPredicate,
    RectanglePredicate,
    UnionOfPredicates,
)


def make_hits():
    # Each row is a synthetic alignment with the relevant fields populated.
    return pd.DataFrame([
        # high identity, high coverage on both sides
        {"query": "a", "target": "b", "fident": 0.95, "qcov": 0.95, "tcov": 0.95},
        # high identity, partial coverage (Case A: deep partial copy)
        {"query": "a", "target": "c", "fident": 0.95, "qcov": 0.90, "tcov": 0.30},
        # low identity, high coverage (Case B: distant whole-protein homology)
        {"query": "a", "target": "d", "fident": 0.32, "qcov": 0.85, "tcov": 0.85},
        # low identity, low coverage (no signal)
        {"query": "a", "target": "e", "fident": 0.30, "qcov": 0.20, "tcov": 0.20},
        # mid identity, mid coverage
        {"query": "a", "target": "f", "fident": 0.55, "qcov": 0.55, "tcov": 0.55},
    ])


def test_rectangle_cov_mode_1_target_only():
    pred = RectanglePredicate(min_pident=0.30, min_coverage=0.80, cov_mode=1)
    hits = make_hits()
    flagged = list(hits[pred(hits)]["target"])
    # cov_mode=1 looks at tcov only: a-b (0.95), a-d (0.85). a-c has tcov=0.30, fails.
    assert flagged == ["b", "d"]


def test_rectangle_cov_mode_0_both_directions():
    pred = RectanglePredicate(min_pident=0.30, min_coverage=0.80, cov_mode=0)
    hits = make_hits()
    flagged = list(hits[pred(hits)]["target"])
    # cov_mode=0 requires both qcov and tcov >= 0.80: a-b, a-d.
    assert flagged == ["b", "d"]


def test_rectangle_high_identity_partial_coverage_rule():
    # Rule 3 from the design: catches deep-partial copies that rule 1 misses.
    pred = RectanglePredicate(min_pident=0.90, min_coverage=0.30, cov_mode=0)
    hits = make_hits()
    flagged = list(hits[pred(hits)]["target"])
    # a-b (qcov 0.95, tcov 0.95, pident 0.95): passes. a-c (tcov 0.30, pident 0.95): passes.
    assert flagged == ["b", "c"]


def test_rectangle_invalid_cov_mode_raises():
    pred = RectanglePredicate(min_pident=0.3, min_coverage=0.8, cov_mode=5)
    import pytest
    with pytest.raises(ValueError):
        pred(make_hits())


def test_hyperbola_predicate():
    pred = HyperbolaPredicate(x=0.24, cov_mode=1)
    hits = make_hits()
    flagged = list(hits[pred(hits)]["target"])
    # tcov * pident: b 0.9025, c 0.285, d 0.272, e 0.06, f 0.3025. All except e pass 0.24.
    assert flagged == ["b", "c", "d", "f"]


def test_hyperbola_cov_mode_0_uses_min_of_qcov_tcov():
    pred = HyperbolaPredicate(x=0.50, cov_mode=0)
    hits = make_hits()
    # min(qcov, tcov) * pident: b 0.9025, c 0.285, d 0.272, e 0.06, f 0.3025.
    flagged = list(hits[pred(hits)]["target"])
    assert flagged == ["b"]


def test_hyperbola_cov_mode_symmetric_uses_max_of_qcov_tcov():
    pred = HyperbolaPredicate(x=0.24, cov_mode=COV_MODE_SYMMETRIC)
    hits = make_hits()
    # max(qcov, tcov) * fident: b 0.9025, c 0.855, d 0.272, e 0.06, f 0.3025.
    # a-c now passes on the *query* side, which cov_mode=1 (tcov 0.30) barely also did;
    # the point is the value no longer depends on which chain was the query.
    flagged = list(hits[pred(hits)]["target"])
    assert flagged == ["b", "c", "d", "f"]


def test_hyperbola_symmetric_mode_is_orientation_invariant():
    # A short query aligned over a long target: near-identical, but coverage is
    # lopsided. This is the shape that made clustering and leak detection
    # disagree (antibody rows, PDB 9j*), so it is the case worth pinning.
    hits = pd.DataFrame([
        {"query": "a", "target": "b", "fident": 0.95, "qcov": 0.90, "tcov": 0.10},
    ])
    swapped = hits.rename(columns={"qcov": "tcov", "tcov": "qcov"})
    pred = HyperbolaPredicate(x=0.24, cov_mode=COV_MODE_SYMMETRIC)
    assert list(pred(hits)) == list(pred(swapped))
    # cov_mode=1 is *not* invariant — this is the inconsistency the mode fixes.
    asym = HyperbolaPredicate(x=0.24, cov_mode=1)
    assert list(asym(hits)) != list(asym(swapped))


def test_hyperbola_min_fident_floor_for_redundancy_screening():
    hits = make_hits()
    pred = HyperbolaPredicate(x=0.24, cov_mode=COV_MODE_SYMMETRIC, min_fident=0.80)
    flagged = list(hits[pred(hits)]["target"])
    # Only the near-identical pairs survive the identity floor: b (0.95), c (0.95).
    assert flagged == ["b", "c"]


def test_symmetric_cov_mode_rejected_nowhere_and_bad_mode_raises():
    import pytest

    with pytest.raises(ValueError, match="Unsupported cov_mode"):
        HyperbolaPredicate(x=0.24, cov_mode=3)(make_hits())


def test_union_of_predicates_or_logic():
    rule_loose = RectanglePredicate(0.30, 0.80, cov_mode=1)
    rule_partial = RectanglePredicate(0.90, 0.30, cov_mode=0)
    pred = UnionOfPredicates(predicates=(rule_loose, rule_partial))
    hits = make_hits()
    flagged = list(hits[pred(hits)]["target"])
    # rule_loose: b, d. rule_partial: b, c. Union: b, c, d.
    assert flagged == ["b", "c", "d"]


def test_union_of_empty_predicates():
    pred = UnionOfPredicates()
    hits = make_hits()
    assert pred(hits).sum() == 0


def test_predicate_roundtrip_rectangle_via_cli_loader():
    from ppisplit.cli import _load_predicate_from_params

    params = {"post_filter": {
        "kind": "rectangle", "min_seq_id": 0.3, "coverage": 0.8, "cov_mode": 1,
    }}
    pred = _load_predicate_from_params(params)
    assert isinstance(pred, RectanglePredicate)
    assert pred.min_pident == 0.3 and pred.min_coverage == 0.8 and pred.cov_mode == 1


def test_predicate_roundtrip_hyperbola_via_cli_loader():
    from ppisplit.cli import _load_predicate_from_params

    params = {"post_filter": {"kind": "hyperbola", "x": 0.30, "cov_mode": 1}}
    pred = _load_predicate_from_params(params)
    assert isinstance(pred, HyperbolaPredicate)
    assert pred.x == 0.30 and pred.cov_mode == 1


def test_predicate_loader_rejects_unknown_kind():
    import pytest

    from ppisplit.cli import _load_predicate_from_params

    with pytest.raises(ValueError, match="Unknown predicate kind"):
        _load_predicate_from_params({"post_filter": {"kind": "circle"}})
