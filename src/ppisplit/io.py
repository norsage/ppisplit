"""I/O for annotations, PDB files, and FASTAs.

Three sequence sources are available; ``seqres-trimmed`` is the default.

``seqres-trimmed``
    The SEQRES slice from the first to the last resolved residue of a chain.
    Unresolved termini, purification tags among them, stay out; unresolved
    internal loops are kept, so a loop that is ordered in one entry and
    disordered in another does not change the comparison.

``resolved``
    Amino-acid residues present in model 0 of the deposited structure, with
    internal gaps joined. Kept to reproduce older verdicts.

``seqres``
    Entity-level full sequences (gemmi ``entity.full_sequence``), including
    signal peptides, purification tags and unresolved loops. Kept to
    reproduce older verdicts.

Why full SEQRES is not used: unresolved tags invent similarity. 1,290 of
8,515 chains in the audit universe carry a purification tag motif and 94.7%
of those instances are unresolved, so e.g. ``9f6o`` scores 0.909 against the
anti-myc Fab peptide ``2or9_P`` under SEQRES and 0.044 under resolved. Unresolved
residues also inflate the coverage denominator: ``6lsa_F ~ 4myw_A`` scores
0.208 under SEQRES and 0.263 under resolved, on an identical alignment.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path

import gemmi
import pandas as pd

from ppisplit.types import ChainKey, ComplexInfo


REQUIRED_COLUMNS = ("uid", "receptor_chains", "ligand_chains")

SEQUENCE_SOURCES = ("seqres-trimmed", "resolved", "seqres")
DEFAULT_SEQUENCE_SOURCE = "seqres-trimmed"

MIN_CHAIN_LENGTH = 3


def parse_chain_ids(chain_str: str) -> tuple[str, ...]:
    """Parse a packed chain-id string like 'DE' into ('D', 'E')."""
    return tuple(str(chain_str))


def load_annotations(*csv_paths: Path) -> list[ComplexInfo]:
    """Load and concatenate annotation CSVs into ComplexInfo records.

    Deduplicates by uid (keeping the first occurrence).
    """
    dfs: list[pd.DataFrame] = []
    for p in csv_paths:
        df = pd.read_csv(p)
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' missing in {p}")
        dfs.append(df[list(REQUIRED_COLUMNS)])

    merged = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="uid", keep="first")
    return [
        ComplexInfo(
            uid=str(row["uid"]),
            receptor_chains=parse_chain_ids(str(row["receptor_chains"])),
            ligand_chains=parse_chain_ids(str(row["ligand_chains"])),
        )
        for _, row in merged.iterrows()
    ]


def _seqres_sequences(st: gemmi.Structure) -> dict[str, str]:
    """Entity-level full sequences (includes unresolved residues and tags)."""
    sequences: dict[str, str] = {}
    for entity in st.entities:
        if entity.polymer_type.name == "PeptideL":
            seq = gemmi.one_letter_code(entity.full_sequence)
            if seq:
                sequences[entity.name] = seq
    return sequences


def _one_letter(name: str) -> str | None:
    """One-letter code of an amino acid; microheterogeneity takes the first alternative."""
    info = gemmi.find_tabulated_residue(name.split(",")[0])
    if info is not None and info.is_amino_acid():
        return info.one_letter_code.upper()
    return None


def _seqres_trimmed_sequences(st: gemmi.Structure) -> dict[str, str]:
    """SEQRES from the first to the last resolved residue, keyed by chain id.

    Resolved residues are placed on the entity sequence by gemmi's
    ``label_seq`` (read from mmCIF, aligned to SEQRES for PDB files). The
    slice keeps unresolved internal loops and drops the termini that were
    never observed, purification tags included. A chain whose residues
    cannot be placed (no entity sequence, no ``label_seq``) falls back to its
    resolved sequence. Chains with no amino acid are absent.
    """
    st.assign_label_seq_id(force=False)
    sequences: dict[str, str] = {}
    for chain in st[0]:
        residues = [r for r in chain if _one_letter(r.name)]
        if not residues:
            continue
        resolved = "".join(_one_letter(r.name) for r in residues)
        polymer = chain.get_polymer()
        entity = st.get_entity_of(polymer) if len(polymer) else None
        placed = [r.label_seq for r in residues if r.label_seq is not None and r.label_seq > 0]
        if entity is None or not entity.full_sequence or not placed:
            sequences[chain.name] = resolved
            continue
        full = entity.full_sequence
        trimmed = (_one_letter(name) for name in full[min(placed) - 1 : max(placed)])
        sequences[chain.name] = "".join(c for c in trimmed if c)
    return sequences


def _resolved_sequences(st: gemmi.Structure) -> dict[str, str]:
    """Amino-acid residues present in model 0, keyed by chain id.

    Waters, nucleotides and small-molecule ligands are excluded by the
    ``is_amino_acid`` check, so no separate ligand-stripping pass is needed.
    Modified residues resolve to their parent one-letter code via gemmi's
    chemical-component table; anything unrecognised is skipped.
    """
    sequences: dict[str, str] = {}
    for chain in st[0]:
        letters: list[str] = []
        for residue in chain:
            info = gemmi.find_tabulated_residue(residue.name)
            if info is not None and info.is_amino_acid():
                letters.append(info.one_letter_code.upper())
        if letters:
            sequences[chain.name] = "".join(letters)
    return sequences


def extract_sequences_from_pdb(
    pdb_path: Path,
    *,
    sequence_source: str = DEFAULT_SEQUENCE_SOURCE,
    fallback_to_chain: bool = True,
) -> dict[str, str]:
    """Read a PDB/mmCIF and return {chain_id: amino_acid_sequence}.

    ``sequence_source='resolved'`` (default) returns the residues observed in
    model 0. A chain with no observed amino acid (DNA, or an all-ligand chain)
    is simply absent — there is no protein sequence to compare, and supplying
    the SEQRES one instead would be the fabrication this source exists to
    avoid. :func:`write_fasta` reports any annotated chain that comes back
    empty, because a silently missing chain is a silently clean verdict.

    ``sequence_source='seqres'`` returns entity-level full sequences, matching
    alpine's ``get_full_sequences``. With ``fallback_to_chain=True``, chains
    absent from any PeptideL entity (e.g. short peptides not classified as
    polymers) are read from the model instead.

    Only sequences of at least :data:`MIN_CHAIN_LENGTH` residues are kept on
    the fallback paths.
    """
    if sequence_source not in SEQUENCE_SOURCES:
        raise ValueError(
            f"sequence_source must be one of {SEQUENCE_SOURCES}, got {sequence_source!r}"
        )

    st = gemmi.read_structure(str(pdb_path))
    st.add_entity_types()
    st.assign_subchains()
    st.ensure_entities()

    if sequence_source == "seqres-trimmed":
        st.deduplicate_entities()
        return _seqres_trimmed_sequences(st)
    if sequence_source == "resolved":
        return _resolved_sequences(st)

    sequences = _seqres_sequences(st)
    if fallback_to_chain:
        for chain in st[0]:
            if chain.name in sequences:
                continue
            three_letter_seq = [residue.name for residue in chain if residue.name]
            if not three_letter_seq:
                continue
            seq = gemmi.one_letter_code(three_letter_seq)
            if seq and len(seq) >= MIN_CHAIN_LENGTH:
                sequences[chain.name] = seq

    return sequences


def write_fasta(
    complexes: Iterable[ComplexInfo],
    pdb_dir: Path,
    output_path: Path,
    *,
    sequence_source: str = DEFAULT_SEQUENCE_SOURCE,
) -> int:
    """Extract chain sequences from PDB files and write a FASTA.

    Each FASTA record is keyed by '{uid}_{chain_id}'. Returns the number
    of sequences written. Exits non-zero if any required PDB is missing.

    An annotated chain that yields no sequence is reported, not skipped
    quietly: a chain absent from the FASTA can never match anything, so it
    turns into a clean leak verdict for free.
    """
    n_written = 0
    missing: list[str] = []
    empty_chains: list[str] = []
    with open(output_path, "w") as f:
        for c in complexes:
            uid = c.uid.lower()
            pdb_path = pdb_dir / f"{uid}.pdb"
            if not pdb_path.exists():
                pdb_path = pdb_dir / f"{uid}.cif"
            if not pdb_path.exists():
                missing.append(uid)
                continue

            try:
                sequences = extract_sequences_from_pdb(
                    pdb_path, sequence_source=sequence_source
                )
            except Exception as e:
                print(f"  Warning: failed to parse {pdb_path}: {e}", file=sys.stderr)
                continue

            wanted = set(c.receptor_chains) | set(c.ligand_chains)
            for chain_id in sorted(wanted):
                if chain_id in sequences:
                    f.write(f">{uid}_{chain_id}\n{sequences[chain_id]}\n")
                    n_written += 1
                else:
                    empty_chains.append(f"{uid}_{chain_id}")
                    print(
                        f"  Warning: chain {chain_id} has no {sequence_source} "
                        f"sequence in {pdb_path.name}",
                        file=sys.stderr,
                    )

    if empty_chains:
        print(
            f"  WARNING: {len(empty_chains)} annotated chain(s) produced no sequence "
            f"and cannot match anything: {empty_chains}",
            file=sys.stderr,
        )
    if missing:
        print(f"  ERROR: {len(missing)} PDB files not found: {missing}", file=sys.stderr)
        sys.exit(1)
    return n_written


def read_fasta_keys(fasta_path: Path) -> set[ChainKey]:
    """Return the set of FASTA record headers (without the '>')."""
    keys: set[ChainKey] = set()
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                keys.add(line[1:].strip())
    return keys


def count_fasta_records(fasta_path: Path) -> int:
    """Number of records in a FASTA — used to size mmseqs's --max-seqs."""
    with open(fasta_path) as f:
        return sum(1 for line in f if line.startswith(">"))


def chain_role_map(complexes: Iterable[ComplexInfo]) -> dict[ChainKey, str]:
    """Build {chain_key: 'receptor' | 'ligand'} for the given complexes."""
    role: dict[ChainKey, str] = {}
    for c in complexes:
        for ch in c.receptor_chains:
            role[f"{c.uid}_{ch}"] = "receptor"
        for ch in c.ligand_chains:
            role[f"{c.uid}_{ch}"] = "ligand"
    return role
