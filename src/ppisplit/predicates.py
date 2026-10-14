"""Edge predicates: turn a raw hit table into similarity edges.

A predicate is any callable ``(hits: DataFrame) -> Series[bool]`` that
selects the rows whose (q, t) pair should be treated as a similarity
edge. Both clustering and leak detection consume the same predicate.

Concrete predicates are dataclasses so they can be serialized to
``params.json`` and reconstructed for downstream steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd

# ppisplit extension to mmseqs's cov-mode numbering: ``max(qcov, tcov)``.
# Negative so it can never collide with an mmseqs mode (0-5).
#
# Why it exists: clustering unions edges symmetrically, while leak detection
# only ever scored the (test=query) orientation. With an orientation-*dependent*
# coverage term (mode 1 = tcov) the two paths disagree — the same pair is an
# edge for clustering but not a leak, which is how antibody rows (PDB 9j*) came out
# clean as a test set while clustering treated them as related. ``max(qcov,
# tcov)`` is orientation invariant, so one predicate governs both.
# It also catches the coverage-asymmetric case a min/tcov rule hides: a fragment
# aligned over a full-length train chain (PDB `4yl6`, fident 1.000 on both
# partners). See the README ("Three invariants").
COV_MODE_SYMMETRIC = -1


def _coverage(hits: pd.DataFrame, cov_mode: int) -> pd.Series:
    """Coverage term selected by ``cov_mode`` (mmseqs numbering + extension)."""
    if cov_mode == COV_MODE_SYMMETRIC:
        return hits[["qcov", "tcov"]].max(axis=1)
    if cov_mode == 0:
        return hits[["qcov", "tcov"]].min(axis=1)
    if cov_mode == 1:
        return hits["tcov"]
    if cov_mode == 2:
        return hits["qcov"]
    raise ValueError(
        f"Unsupported cov_mode {cov_mode}; supported: -1 (max), 0 (min/both), "
        "1 (tcov), 2 (qcov). Modes 3-5 are mmseqs length-similarity prefilters "
        "and are not implemented."
    )


class EdgePredicate(Protocol):
    """A boolean filter over the columns of a hit table."""

    def __call__(self, hits: pd.DataFrame) -> pd.Series:
        ...


@dataclass(frozen=True)
class RectanglePredicate:
    """Single (fident, coverage) rectangle.

    Identity is read from the ``fident`` column (0-1 fraction). cov_mode
    follows mmseqs2, plus one ppisplit extension:
      -1 -> max(qcov, tcov) >= coverage  (orientation invariant)
      0 -> qcov >= coverage AND tcov >= coverage
      1 -> tcov >= coverage
      2 -> qcov >= coverage
    """

    min_pident: float  # threshold on fident (0-1), naming preserved for CLI familiarity
    min_coverage: float
    cov_mode: int = 0

    def __call__(self, hits: pd.DataFrame) -> pd.Series:
        return (hits["fident"] >= self.min_pident) & (
            _coverage(hits, self.cov_mode) >= self.min_coverage
        )


@dataclass(frozen=True)
class HyperbolaPredicate:
    """fident * coverage >= x. Identity is read from the ``fident`` column (0-1).

    ``cov_mode=-1`` (``COV_MODE_SYMMETRIC``, the canonical setting) uses
    ``max(qcov, tcov)`` so the predicate gives the same answer whichever chain
    of a pair is the query — required for clustering and leak detection to
    agree. Optionally combined with ``min_fident``, an identity floor used for
    test-internal *redundancy* screening (not for leak detection, where the
    floor would let a distant-but-full-length homolog through).
    """

    x: float
    cov_mode: int = COV_MODE_SYMMETRIC
    min_fident: float = 0.0

    def __call__(self, hits: pd.DataFrame) -> pd.Series:
        mask = hits["fident"] * _coverage(hits, self.cov_mode) >= self.x
        if self.min_fident > 0.0:
            mask &= hits["fident"] >= self.min_fident
        return mask


@dataclass(frozen=True)
class UnionOfPredicates:
    """OR of multiple predicates: an edge if *any* sub-predicate fires."""

    predicates: tuple[EdgePredicate, ...] = field(default_factory=tuple)

    def __call__(self, hits: pd.DataFrame) -> pd.Series:
        if not self.predicates:
            return pd.Series([False] * len(hits), index=hits.index)
        mask = self.predicates[0](hits)
        for p in self.predicates[1:]:
            mask = mask | p(hits)
        return mask
