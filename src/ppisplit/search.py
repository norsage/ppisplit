"""mmseqs2 wrapper.

The search produces the hit table consumed by both clustering and leak
detection. Threshold logic lives entirely in :mod:`ppisplit.predicates` and is
applied post-hoc.

**The search parameters below are recall parameters. They must never decide.**
Every one of them is set so that it cannot filter a pair the post-filter
predicate would have admitted. This is not a style preference — it is the fix
for a measured defect:

* ``-e 0.1`` scales with database size, so verdicts depended on what else
  happened to be in the universe: 73,419 qualifying-edge disagreements between
  a 4,574- and an 8,515-chain universe, measured on chains present in both.
  ``-e 10000`` was still binding (45 unstable qualifying edges). ``-e 1e9``,
  ``1e15`` and ``1e30`` return identical hit counts — the list saturates — and
  at ``1e30`` there are **0 unstable qualifying edges** across the two
  universes.
* ``--min-seq-id 0.30`` is unsound regardless of universe: the canonical
  predicate admits ``fident >= 0.24``, so pairs in [0.24, 0.30) were discarded
  before the predicate ever saw them.
* ``--max-seqs 1000`` truncates hub chains. It is derived from the target
  database size at runtime instead, so it can never bind.

If you change one of these, you are changing the protocol, not tuning it.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from ppisplit.io import count_fasta_records

SEARCH_SETTINGS: dict[str, float | int | None] = {
    "min_seq_id": 0.0,   # no identity floor; the post-filter admits fident >= 0.24
    "coverage": 0.0,     # no coverage filter at search time; post-filter handles it
    "cov_mode": 0,       # mmseqs default
    "evalue": 1e30,      # above saturation; E-value must not scale verdicts with DB size
    "sensitivity": 7.5,  # max sensitivity; cheap on ~thousands of chains
    "max_seqs": None,    # None -> derived from |target DB| so it can never truncate
}


HIT_COLUMNS = (
    "query", "target", "fident", "pident", "alnlen",
    "qlen", "tlen", "qcov", "tcov", "evalue", "bits",
)


def check_mmseqs2() -> None:
    """Exit non-zero if mmseqs2 is not on PATH."""
    if shutil.which("mmseqs") is None:
        print(
            "Error: mmseqs2 not found on PATH. Install via "
            "'conda install -c bioconda mmseqs2' or build from source.",
            file=sys.stderr,
        )
        sys.exit(1)


def resolve_max_seqs(
    settings: dict[str, float | int | None], target_fasta: Path
) -> int:
    """``--max-seqs`` for this search: the configured value, or |target DB|.

    Deriving it means the hit list is never truncated, so adding sequences to
    the database cannot remove an edge that was there before.
    """
    configured = settings.get("max_seqs")
    if configured is not None:
        return int(configured)
    return max(1, count_fasta_records(target_fasta))


def _run_easy_search(
    query_fasta: Path,
    target_fasta: Path,
    raw_tsv: Path,
    tmp_dir: Path,
    settings: dict[str, float | int | None],
) -> pd.DataFrame:
    """One ``mmseqs easy-search`` invocation; returns the raw hit table."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mmseqs", "easy-search",
        str(query_fasta), str(target_fasta), str(raw_tsv), str(tmp_dir),
        "--min-seq-id", str(settings["min_seq_id"]),
        "-c", str(settings["coverage"]),
        "--cov-mode", str(settings["cov_mode"]),
        "-e", str(settings["evalue"]),
        "-s", str(settings["sensitivity"]),
        "--max-seqs", str(resolve_max_seqs(settings, target_fasta)),
        "--format-output", ",".join(HIT_COLUMNS),
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"  mmseqs2 stderr:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"mmseqs2 search failed (exit {result.returncode})")

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if raw_tsv.stat().st_size == 0:
        df = pd.DataFrame(columns=list(HIT_COLUMNS))
    else:
        df = pd.read_csv(raw_tsv, sep="\t", names=list(HIT_COLUMNS))
    raw_tsv.unlink(missing_ok=True)
    return df


def run_mmseqs_search_allvall(
    fasta_path: Path,
    output_dir: Path,
    settings: dict[str, float | int | None] | None = None,
) -> pd.DataFrame:
    """Run mmseqs easy-search self-vs-self on ``fasta_path``.

    Returns a DataFrame with columns from :data:`HIT_COLUMNS`, with
    self-hits removed and edges symmetrized (every (q, t) hit also
    appears as (t, q)).

    Use this for the questions whose query and target databases are the same
    set: fold clustering (train vs train) and within-set redundancy.
    """
    check_mmseqs2()

    s: dict[str, float | int | None] = {**SEARCH_SETTINGS, **(settings or {})}
    df = _run_easy_search(
        fasta_path, fasta_path,
        output_dir / "hits_raw.tsv", output_dir / "mmseqs_tmp", s,
    )
    if df.empty:
        return df
    df = df[df["query"] != df["target"]].reset_index(drop=True)
    return symmetrize_hits(df)


def run_mmseqs_search_pair(
    query_fasta: Path,
    target_fasta: Path,
    output_dir: Path,
    settings: dict[str, float | int | None] | None = None,
) -> pd.DataFrame:
    """Bidirectional search of ``query_fasta`` against ``target_fasta``.

    mmseqs is asymmetric: it reports genuinely different alignments for the two
    directions on ~15% of pairs. Both directions are therefore run and merged,
    with the reverse run's columns swapped so that ``query`` is always a record
    of ``query_fasta``. Where both directions produced a row for the same pair
    the forward one is kept.

    This is the *purpose-split* search: one (query, target) database pair per
    question, never pooled. Adding a test set creates a new comparison; it does
    not enlarge an existing comparison's database, so no verdict can move
    because of an unrelated set.
    """
    check_mmseqs2()

    s: dict[str, float | int | None] = {**SEARCH_SETTINGS, **(settings or {})}
    fwd = _run_easy_search(
        query_fasta, target_fasta,
        output_dir / "hits_fwd_raw.tsv", output_dir / "mmseqs_tmp_fwd", s,
    )
    rev = _run_easy_search(
        target_fasta, query_fasta,
        output_dir / "hits_rev_raw.tsv", output_dir / "mmseqs_tmp_rev", s,
    )

    rev = rev.rename(columns={
        "query": "target", "target": "query",
        "qlen": "tlen", "tlen": "qlen",
        "qcov": "tcov", "tcov": "qcov",
    })
    combined = pd.concat([fwd, rev[list(HIT_COLUMNS)]], ignore_index=True)
    # No self-hit removal here: the two databases are different sets, so a query and a
    # target record sharing an identifier are one chain present in both — a leak, not
    # a self-match. Dropping them hid such chains from the predicate.
    return (
        combined.drop_duplicates(subset=["query", "target"], keep="first")
        .reset_index(drop=True)
    )


def symmetrize_hits(hits: pd.DataFrame) -> pd.DataFrame:
    """Ensure each (q, t) edge has a matching (t, q) row.

    mmseqs may report A->B without B->A even when both directions would
    pass thresholds. We swap query/target columns and append the missing
    direction. Coverage and length columns are swapped accordingly.
    """
    swap = hits.rename(columns={
        "query": "target", "target": "query",
        "qlen": "tlen", "tlen": "qlen",
        "qcov": "tcov", "tcov": "qcov",
    })
    combined = pd.concat([hits, swap[hits.columns.tolist()]], ignore_index=True)
    return combined.drop_duplicates(subset=["query", "target"]).reset_index(drop=True)
