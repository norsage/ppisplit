"""A record present in both databases of a pair search is a hit, not a self-hit.

In a purpose-split search the query and target databases are different sets, so a
query record and a target record that share an identifier (``5vnw_A`` in a test set and
in a training set) are the same chain found in both - the strongest possible leak.
Dropping such rows as self-matches hid these chains from the predicate path.
"""

import shutil
from pathlib import Path

import pytest

from ppisplit.search import run_mmseqs_search_pair

SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWNAIGLNAMLDIMAQAAG"


@pytest.mark.skipif(shutil.which("mmseqs") is None, reason="mmseqs not on PATH")
def test_shared_identifier_across_databases_is_kept(tmp_path: Path):
    query = tmp_path / "query.fasta"
    target = tmp_path / "target.fasta"
    query.write_text(f">5vnw_A\n{SEQ}\n")
    target.write_text(f">5vnw_A\n{SEQ}\n>1abc_B\nGSHMLEDPVDAFQ\n")

    hits = run_mmseqs_search_pair(query, target, tmp_path / "out")

    same = hits[(hits["query"] == "5vnw_A") & (hits["target"] == "5vnw_A")]
    assert len(same) == 1
    assert same["fident"].iloc[0] == pytest.approx(1.0)
