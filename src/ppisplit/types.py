"""Core data types."""

from __future__ import annotations

from dataclasses import dataclass


ChainKey = str  # "{uid}_{chain_id}"


@dataclass(frozen=True)
class ComplexInfo:
    """A protein-protein complex with explicit receptor/ligand chain partitioning."""

    uid: str
    receptor_chains: tuple[str, ...]
    ligand_chains: tuple[str, ...]

    def chain_keys(self) -> tuple[ChainKey, ...]:
        return tuple(f"{self.uid}_{ch}" for ch in (*self.receptor_chains, *self.ligand_chains))

    def receptor_keys(self) -> tuple[ChainKey, ...]:
        return tuple(f"{self.uid}_{ch}" for ch in self.receptor_chains)

    def ligand_keys(self) -> tuple[ChainKey, ...]:
        return tuple(f"{self.uid}_{ch}" for ch in self.ligand_chains)


@dataclass(frozen=True)
class Hit:
    """A single sequence-similarity alignment between two chains."""

    query: ChainKey
    target: ChainKey
    pident: float
    alnlen: int
    qlen: int
    tlen: int
    qcov: float
    tcov: float
    evalue: float
