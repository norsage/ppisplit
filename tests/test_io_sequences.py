"""Sequence extraction: 'seqres-trimmed' is the protocol; 'resolved' and 'seqres' are kept for older verdicts.

The synthetic fixtures pin the semantics; the fixtures built from real PDBs pin
the two worked examples from the protocol revision, using examples/pdb/.
"""

from pathlib import Path

import pytest

from ppisplit.io import (
    DEFAULT_SEQUENCE_SOURCE,
    extract_sequences_from_pdb,
    write_fasta,
)
from ppisplit.types import ComplexInfo


PDB_DIR = Path(__file__).resolve().parents[1] / "examples" / "pdb"


# ---------------------------------------------------------------------------
# Synthetic structures
# ---------------------------------------------------------------------------

def _atom_line(serial, resname, chain, resseq):
    return (
        f"ATOM  {serial:>5d}  CA  {resname:>3s} {chain}{resseq:>4d}    "
        f"{0.0:8.3f}{0.0:8.3f}{float(serial):8.3f}  1.00  0.00           C"
    )


def write_pdb(path: Path, chains: dict[str, list[tuple[int, str]]], seqres=None) -> Path:
    """Write a minimal CA-only PDB. ``chains`` maps chain id -> [(resseq, resname)]."""
    lines: list[str] = []
    if seqres:
        for chain_id, resnames in seqres.items():
            for i in range(0, len(resnames), 13):
                chunk = resnames[i : i + 13]
                lines.append(
                    f"SEQRES {i // 13 + 1:>3d} {chain_id} {len(resnames):>4d}  "
                    + " ".join(f"{r:>3s}" for r in chunk)
                )
    serial = 1
    for chain_id, residues in chains.items():
        for resseq, resname in residues:
            lines.append(_atom_line(serial, resname, chain_id, resseq))
            serial += 1
        lines.append(f"TER   {serial:>5d}      {residues[-1][1]:>3s} {chain_id}{residues[-1][0]:>4d}")
        serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


ALA, GLY, SER = "ALA", "GLY", "SER"


def test_resolved_skips_unresolved_tag_residues(tmp_path):
    """A His-tag present only in SEQRES must not be part of the compared sequence.

    94.7% of tag-motif instances in the audit universe are unresolved. Under
    SEQRES they produce spurious matches between unrelated constructs.
    """
    resolved = [(i, ALA) for i in range(1, 11)]
    seqres = {"A": ["HIS"] * 6 + [ALA] * 10}
    pdb = write_pdb(tmp_path / "tag.pdb", {"A": resolved}, seqres=seqres)

    assert extract_sequences_from_pdb(pdb, sequence_source="resolved")["A"] == "A" * 10
    assert extract_sequences_from_pdb(pdb, sequence_source="seqres")["A"] == "H" * 6 + "A" * 10


def test_resolved_concatenates_across_an_internal_gap(tmp_path):
    """A disordered loop is absent from the resolved sequence, not padded.

    Coverage denominators must count observed residues only — that is exactly
    what makes a fragment-vs-full-length alignment score honestly.
    """
    residues = [(i, ALA) for i in range(1, 6)] + [(i, GLY) for i in range(20, 25)]
    seqres = {"A": [ALA] * 5 + [SER] * 14 + [GLY] * 5}
    pdb = write_pdb(tmp_path / "gap.pdb", {"A": residues}, seqres=seqres)

    assert extract_sequences_from_pdb(pdb, sequence_source="resolved")["A"] == "AAAAAGGGGG"
    assert len(extract_sequences_from_pdb(pdb, sequence_source="seqres")["A"]) == 24


def test_seqres_trimmed_restores_an_internal_loop(tmp_path):
    """Trimmed SEQRES keeps the residues between the first and last resolved ones."""
    residues = [(i, ALA) for i in range(1, 6)] + [(i, GLY) for i in range(20, 25)]
    seqres = {"A": [ALA] * 5 + [SER] * 14 + [GLY] * 5}
    pdb = write_pdb(tmp_path / "gap.pdb", {"A": residues}, seqres=seqres)

    assert extract_sequences_from_pdb(pdb, sequence_source="seqres-trimmed")["A"] == "A" * 5 + "S" * 14 + "G" * 5


def test_seqres_trimmed_drops_unresolved_termini(tmp_path):
    """An unresolved N-terminal tag and C-terminal tail are trimmed off."""
    residues = [(i, ALA) for i in range(7, 17)]
    seqres = {"A": ["HIS"] * 6 + [ALA] * 10 + [GLY] * 4}
    pdb = write_pdb(tmp_path / "tag.pdb", {"A": residues}, seqres=seqres)

    assert extract_sequences_from_pdb(pdb, sequence_source="seqres-trimmed")["A"] == "A" * 10


def test_seqres_trimmed_falls_back_to_resolved_without_seqres(tmp_path):
    pdb = write_pdb(tmp_path / "noseqres.pdb", {"A": [(1, ALA), (2, GLY), (5, SER)]})
    assert extract_sequences_from_pdb(pdb, sequence_source="seqres-trimmed")["A"] == "AGS"


def test_resolved_excludes_waters_and_ligands(tmp_path):
    pdb_text = (
        _atom_line(1, ALA, "A", 1) + "\n"
        + _atom_line(2, GLY, "A", 2) + "\n"
        + _atom_line(3, SER, "A", 3) + "\n"
        + "HETATM    4  O   HOH A 101       0.000   0.000   0.000  1.00  0.00           O\n"
        + "HETATM    5  PG  ANP A 102       0.000   0.000   0.000  1.00  0.00           P\n"
        + "END\n"
    )
    pdb = tmp_path / "ligand.pdb"
    pdb.write_text(pdb_text)
    assert extract_sequences_from_pdb(pdb, sequence_source="resolved")["A"] == "AGS"


def test_short_peptide_antigen_survives(tmp_path):
    """An 11-residue peptide antigen is a real interface — never length-filtered.

    `2or9_P`, the anti-myc Fab's peptide antigen, is the one genuine
    tag-motif interface in the audit universe. Dropping short chains would
    delete it, and with it the leak edges it participates in.
    """
    peptide = [(i, ALA) for i in range(1, 12)]
    pdb = write_pdb(tmp_path / "peptide.pdb", {"P": peptide})
    assert extract_sequences_from_pdb(pdb, sequence_source="resolved")["P"] == "A" * 11


def test_chain_with_no_amino_acids_is_reported_not_silently_dropped(capsys, tmp_path):
    """A chain absent from the FASTA can never match — that is a free clean
    verdict, so write_fasta has to say so out loud."""
    pdb_text = (
        _atom_line(1, ALA, "A", 1) + "\n"
        + _atom_line(2, ALA, "A", 2) + "\n"
        + _atom_line(3, ALA, "A", 3) + "\n"
        + "ATOM      4  P    DA  B   1       0.000   0.000   4.000  1.00  0.00           P\n"
        + "ATOM      5  P    DT  B   2       0.000   0.000   5.000  1.00  0.00           P\n"
        + "END\n"
    )
    (tmp_path / "1dna.pdb").write_text(pdb_text)

    sequences = extract_sequences_from_pdb(tmp_path / "1dna.pdb", sequence_source="resolved")
    assert sequences == {"A": "AAA"}

    n = write_fasta(
        [ComplexInfo("1dna", ("A",), ("B",))], tmp_path, tmp_path / "out.fasta"
    )
    assert n == 1
    assert "1dna_B" in capsys.readouterr().err


def test_extract_rejects_an_unknown_source(tmp_path):
    pdb = write_pdb(tmp_path / "x.pdb", {"A": [(1, ALA), (2, ALA), (3, ALA)]})
    with pytest.raises(ValueError):
        extract_sequences_from_pdb(pdb, sequence_source="alphafold")


def test_write_fasta_defaults_to_seqres_trimmed(tmp_path):
    write_pdb(
        tmp_path / "1abc.pdb",
        {"A": [(i, ALA) for i in range(1, 6)], "B": [(i, GLY) for i in range(1, 6)]},
        seqres={"A": ["HIS"] * 6 + [ALA] * 5, "B": [GLY] * 5},
    )
    out = tmp_path / "seqs.fasta"
    n = write_fasta([ComplexInfo("1abc", ("A",), ("B",))], tmp_path, out)
    assert n == 2
    assert DEFAULT_SEQUENCE_SOURCE == "seqres-trimmed"
    assert ">1abc_A\nAAAAA\n" in out.read_text()


# ---------------------------------------------------------------------------
# Worked examples from the protocol revision
# ---------------------------------------------------------------------------

def test_2or9_peptide_antigen_survives_resolved_extraction():
    sequences = extract_sequences_from_pdb(PDB_DIR / "2or9.pdb", sequence_source="resolved")
    assert "P" in sequences, "the anti-myc Fab's peptide antigen must not be dropped"
    assert 5 <= len(sequences["P"]) <= 20


def test_9f6o_myc_tag_is_seqres_only():
    """The tag that made 9f6o look 0.909 identical to 2or9_P is unresolved."""
    myc = "EQKLISEEDL"
    resolved = extract_sequences_from_pdb(PDB_DIR / "9f6o.pdb", sequence_source="resolved")
    trimmed = extract_sequences_from_pdb(PDB_DIR / "9f6o.pdb", sequence_source="seqres-trimmed")
    seqres = extract_sequences_from_pdb(PDB_DIR / "9f6o.pdb", sequence_source="seqres")
    assert any(myc in s for s in seqres.values())
    assert not any(myc in s for s in resolved.values())
    assert not any(myc in s for s in trimmed.values())
