"""The CLI defaults ARE the protocol — pin them.

The canonical rule is ``fident * max(qcov, tcov) >= 0.24`` with a leak requiring
both partners to match (``similarity-mode both``). These tests fail if a default
drifts back to the legacy rectangle rule or to an orientation-dependent coverage
mode. See the README ("Three invariants").
"""

import pytest

from ppisplit.cli import _load_predicate_from_params, _predicate_config_from_args, build_parser
from ppisplit.predicates import COV_MODE_SYMMETRIC, HyperbolaPredicate
from ppisplit.search import SEARCH_SETTINGS


def _cluster_args(*extra):
    return build_parser().parse_args([
        "cluster", "--csv", "a.csv", "--pdb-dir", "pdb", "--output-dir", "out", *extra,
    ])


def test_cluster_defaults_are_the_canonical_predicate():
    args = _cluster_args()
    assert args.predicate == "hyperbola"
    assert args.hyperbola_x == 0.24
    assert args.cov_mode == COV_MODE_SYMMETRIC
    assert args.min_fident == 0.0  # identity floor is redundancy-screening only


def test_default_config_round_trips_to_the_canonical_predicate_object():
    cfg = _predicate_config_from_args(_cluster_args())
    pred = _load_predicate_from_params({"post_filter": cfg})
    assert pred == HyperbolaPredicate(x=0.24, cov_mode=COV_MODE_SYMMETRIC, min_fident=0.0)


@pytest.mark.parametrize("cmd", ["check-leaks", "clean-test-set", "split"])
def test_similarity_mode_defaults_to_both(cmd):
    parser = build_parser()
    required = {
        "check-leaks": ["--train-csv", "t.csv", "--test-csv", "s.csv", "--cluster-dir", "d"],
        "clean-test-set": ["--test-csv", "s.csv", "--train-csv", "t.csv",
                           "--cluster-dir", "d", "--output-dir", "o"],
        "split": ["--csv", "t.csv", "--cluster-dir", "d", "--output", "f.csv",
                  "--mode", "kfold", "--n-folds", "5"],
    }[cmd]
    assert parser.parse_args([cmd, *required]).similarity_mode == "both"


@pytest.mark.parametrize("x", ["0", "-0.1", "1.5"])
def test_out_of_range_hyperbola_x_is_rejected(x):
    with pytest.raises(SystemExit) as exc:
        _predicate_config_from_args(_cluster_args("--hyperbola-x", x))
    assert exc.value.code == 2


def test_sequence_source_defaults_to_seqres_trimmed():
    """Full SEQRES invents similarity through unresolved tags; resolved residues
    make a verdict depend on which entry happened to order a loop. Trimmed SEQRES
    avoids both. The other two stay reachable for older verdicts."""
    assert _cluster_args().sequence_source == "seqres-trimmed"
    assert _cluster_args("--sequence-source", "resolved").sequence_source == "resolved"
    assert _cluster_args("--sequence-source", "seqres").sequence_source == "seqres"


def test_search_is_self_vs_self_unless_a_target_set_is_given():
    """Purpose-split scope: one (query, target) DB pair per question."""
    assert _cluster_args().target_csv == []
    args = _cluster_args("--target-csv", "train.csv", "--target-csv", "train2.csv")
    assert args.target_csv == ["train.csv", "train2.csv"]


def test_recorded_search_settings_cannot_filter_the_predicate():
    """params.json records the search settings; they must stay non-binding."""
    assert SEARCH_SETTINGS["min_seq_id"] == 0.0
    assert SEARCH_SETTINGS["coverage"] == 0.0
    assert SEARCH_SETTINGS["max_seqs"] is None
    assert SEARCH_SETTINGS["evalue"] >= 1e9


def test_rectangle_is_still_reachable_for_historical_numbers():
    cfg = _predicate_config_from_args(
        _cluster_args("--predicate", "rectangle", "--cov-mode", "1")
    )
    assert cfg == {"kind": "rectangle", "min_seq_id": 0.3, "coverage": 0.8, "cov_mode": 1}
