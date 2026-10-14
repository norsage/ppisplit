"""Search parameters are recall parameters — they must never decide a verdict.

The end-to-end property is: two differently-sized universes must give identical
verdicts for the complexes they share. The mmseqs-backed version of that test
is skipped when mmseqs is not on PATH; the settings assertions below hold
unconditionally and are what actually pins the protocol.
"""

import shutil
from pathlib import Path

import pytest

from ppisplit.io import count_fasta_records, load_annotations
from ppisplit.predicates import COV_MODE_SYMMETRIC, HyperbolaPredicate
from ppisplit.search import SEARCH_SETTINGS, resolve_max_seqs, run_mmseqs_search_pair
from ppisplit.types import ComplexInfo


def test_evalue_is_above_saturation():
    """`-e 0.1` scaled with DB size: 73,419 qualifying-edge disagreements
    between a 4,574- and an 8,515-chain universe. `-e 10000` was still binding
    (45 unstable edges). Hit counts saturate above 1e9; 1e30 is safely past it."""
    assert SEARCH_SETTINGS["evalue"] >= 1e9


def test_no_identity_floor_at_search_time():
    """The canonical predicate admits fident >= 0.24, so a 0.30 search-time
    floor discarded pairs before the predicate could evaluate them."""
    assert SEARCH_SETTINGS["min_seq_id"] == 0.0


def test_max_seqs_is_derived_from_the_target_database(tmp_path):
    """A fixed cap truncates hub chains — latent today, binding at scale."""
    assert SEARCH_SETTINGS["max_seqs"] is None

    fasta = tmp_path / "t.fasta"
    fasta.write_text(">a\nAAAA\n>b\nCCCC\n>c\nDDDD\n")
    assert count_fasta_records(fasta) == 3
    assert resolve_max_seqs(SEARCH_SETTINGS, fasta) == 3
    assert resolve_max_seqs({**SEARCH_SETTINGS, "max_seqs": 7}, fasta) == 7


def test_empty_target_database_still_yields_a_legal_max_seqs(tmp_path):
    fasta = tmp_path / "empty.fasta"
    fasta.write_text("")
    assert resolve_max_seqs(SEARCH_SETTINGS, fasta) >= 1


def test_no_coverage_filter_at_search_time():
    assert SEARCH_SETTINGS["coverage"] == 0.0


def test_sensitivity_is_maximum():
    assert SEARCH_SETTINGS["sensitivity"] == 7.5


# ---------------------------------------------------------------------------
# Universe invariance, end to end
# ---------------------------------------------------------------------------

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
PDB_DIR = EXAMPLES / "pdb"

needs_mmseqs = pytest.mark.skipif(
    shutil.which("mmseqs") is None,
    reason="requires mmseqs2 on PATH",
)

# A probe pair plus the example complexes as padding. The probe verdicts must
# not move when the padding is added. This is a smoke test of the end-to-end
# mmseqs path, not a regression test for the E-value defect: at this database
# size it still passes with the old `-e 0.1`. The defect needed thousands of
# chains to show; the SEARCH_SETTINGS assertions above are what pin it.
PROBE = [
    ComplexInfo("4yl6", ("A",), ("B",)),
    ComplexInfo("4y5o", ("A",), ("B",)),
]


@needs_mmseqs
def test_verdicts_are_invariant_to_universe_size(tmp_path):
    from ppisplit.io import write_fasta

    predicate = HyperbolaPredicate(x=0.24, cov_mode=COV_MODE_SYMMETRIC)

    def qualifying_edges(target_complexes, tag):
        out = tmp_path / tag
        out.mkdir()
        query_fasta = out / "q.fasta"
        target_fasta = out / "t.fasta"
        write_fasta(PROBE, PDB_DIR, query_fasta)
        write_fasta(target_complexes, PDB_DIR, target_fasta)
        hits = run_mmseqs_search_pair(query_fasta, target_fasta, out)
        edges = hits[predicate(hits)]
        return set(zip(edges["query"], edges["target"]))

    padding = load_annotations(EXAMPLES / "complexes.csv")
    small = qualifying_edges(PROBE, "small")
    large = qualifying_edges([*PROBE, *padding], "large")

    probe_chains = {k for c in PROBE for k in c.chain_keys()}

    def restrict(edges):
        return {(q, t) for q, t in edges if q in probe_chains and t in probe_chains}

    assert restrict(small) == restrict(large)
    assert restrict(small), "the probe pair must produce at least one qualifying edge"
