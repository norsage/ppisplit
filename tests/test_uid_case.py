"""A uid keeps its case everywhere, and a CSV whose chains match nothing fails loudly.

`write_fasta` used to lower-case uids while `ComplexInfo.chain_keys` did not, so every
command reading the cluster directory found no chains for an upper-case uid and went on
silently with an empty set (task 0001.z9r).
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from ppisplit.cli import require_known_chains
from ppisplit.io import write_fasta
from ppisplit.types import ComplexInfo

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _upper_case_example(tmp_path: Path) -> Path:
    pdb = tmp_path / "pdb"
    pdb.mkdir()
    df = pd.read_csv(EXAMPLES / "complexes.csv")
    for uid in df.uid:
        shutil.copy(EXAMPLES / "pdb" / f"{uid}.pdb", pdb / f"{uid.upper()}.pdb")
    df["uid"] = df.uid.str.upper()
    df.to_csv(tmp_path / "up.csv", index=False)
    return tmp_path


def test_fasta_keeps_the_uid_case(tmp_path):
    root = _upper_case_example(tmp_path)
    out = root / "seqs.fasta"
    write_fasta([ComplexInfo("1A22", ("B",), ("A",))], root / "pdb", out)
    headers = [line[1:] for line in out.read_text().splitlines() if line.startswith(">")]
    assert headers == ["1A22_A", "1A22_B"]


def test_no_known_chains_is_an_error():
    complexes = [ComplexInfo("1A22", ("B",), ("A",))]
    with pytest.raises(SystemExit):
        require_known_chains(complexes, {"1a22_A", "1a22_B"}, "split")


def test_some_known_chains_pass():
    complexes = [ComplexInfo("1A22", ("B",), ("A",)), ComplexInfo("2YBR", ("A",), ("C",))]
    require_known_chains(complexes, {"1A22_A", "1A22_B"}, "split")


@pytest.mark.skipif(shutil.which("mmseqs") is None, reason="mmseqs not on PATH")
@pytest.mark.parametrize("mode", [["--mode", "montecarlo", "--n-splits", "2"],
                                  ["--mode", "kfold", "--n-folds", "2"]])
def test_upper_case_uids_split_like_lower_case(tmp_path, mode):
    root = _upper_case_example(tmp_path)
    cli = [sys.executable, "-m", "ppisplit.cli"]
    subprocess.run([*cli, "cluster", "--csv", root / "up.csv", "--pdb-dir", root / "pdb",
                    "--output-dir", root / "out"], check=True, capture_output=True)
    subprocess.run([*cli, "split", "--csv", root / "up.csv", "--cluster-dir", root / "out",
                    "--output", root / "folds.csv", *mode], check=True, capture_output=True)
    folds = pd.read_csv(root / "folds.csv")
    assert sorted(folds.uid) == ["1A22", "1BVK", "2YBR", "5TZ2"]
