# ppisplit

Sequence-aware cross-validation splitting and leak detection for protein-protein interaction datasets.

## Requirements

- Python 3.12 or newer.
- [MMseqs2](https://github.com/soedinglab/MMseqs2) on your `PATH`. `ppisplit cluster` runs the `mmseqs` binary to build the hit table; the other commands read that table. Install it with `conda install -c bioconda mmseqs2`, `brew install mmseqs2`, or from source, and check with `mmseqs version`.

## Install

```bash
pip install git+https://github.com/norsage/ppisplit.git
```

## Quick start

The repository ships a four-complex example: [`examples/complexes.csv`](examples/complexes.csv) and its structures in [`examples/pdb/`](examples/pdb/) (unmodified RCSB PDB entries, CC0). The tests read the same directory.

```bash
ppisplit cluster --csv examples/complexes.csv --pdb-dir examples/pdb --output-dir out/example
ppisplit summary --cluster-dir out/example --csv examples/complexes.csv
```

## Input format

A dataset is a CSV with one row per complex. Three columns are required and any others are ignored.

| column | meaning |
|---|---|
| `uid` | complex id; the structure is read from `<pdb-dir>/<uid>.pdb`, or `.cif` if there is no `.pdb` |
| `receptor_chains` | chain ids of one partner |
| `ligand_chains` | chain ids of the other partner |

A side with several chains lists them with no separator, so `HL` means chains `H` and `L`. Each chain id must be a single character. Single-chain and multi-chain complexes can sit in the same file; the example has both:

| `uid` | `receptor_chains` | `ligand_chains` | meaning |
|---|---|---|---|
| `1a22` | `B` | `A` | one chain per side |
| `1bvk` | `C` | `AB` | ligand is chains `A` and `B` |
| `5tz2` | `C` | `HL` | ligand is chains `H` and `L` |
| `2ybr` | `AB` | `C` | receptor is chains `A` and `B` |

## How complexes are compared

Each chain is compared on its own. Two chains are similar when `fident * max(qcov, tcov) >= 0.24`, where `fident` is the fraction of identical residues in the MMseqs2 alignment and `qcov`, `tcov` are the fractions of each chain it covers. The threshold is `--hyperbola-x` on `ppisplit cluster`.

One side of a complex matches a side of another complex when any of its chains is similar to any chain on that side. Under `--similarity-mode both`, the default, two complexes are similar when both sides match: receptor with receptor and ligand with ligand, or receptor with ligand and ligand with receptor in case the two datasets label the partners the other way round. For `5tz2`, chain `C` has to match one side of the other complex and `H` or `L` has to match the other side.

The same rule decides which complexes share a fold and which test complexes leak. A test complex also leaks when its `uid` appears in the training set. The modes `receptor-only`, `ligand-only` and `either` are for diagnostics.

## Commands

Run one search per question, with the sets that question is about and nothing else, so adding a dataset never changes the verdict of an existing comparison.

| question | `--csv` (query) | `--target-csv` |
|---|---|---|
| does the test set leak into train? | test set | train sets |
| do two test sets overlap? | test A | test B |
| is a set redundant within itself? | test A | (none) |
| fold clustering | train | (none) |

```bash
# 1a) Cluster, self-vs-self: fold clustering, or within-set redundancy.
ppisplit cluster \
    --csv train.csv \
    --pdb-dir pdbs \
    --output-dir splits/run1

# 1b) Cluster, purpose-split: a test set against the train sets.
ppisplit cluster \
    --csv test.csv \
    --target-csv train.csv \
    --pdb-dir pdbs \
    --output-dir splits/leaks_test_vs_train

# 2) Split: post-filter, build clusters, assign folds.
ppisplit split \
    --csv train.csv \
    --cluster-dir splits/run1 \
    --output splits/run1/folds_mc10.csv \
    --mode montecarlo --n-splits 10 --val-ratio 0.25 \
    --similarity-mode both --seed 42 \
    --test-csv test.csv

# 3) Check-leaks: same hit table, role-aware leak report.
ppisplit check-leaks \
    --train-csv train.csv \
    --test-csv test.csv \
    --cluster-dir splits/leaks_test_vs_train \
    --similarity-mode both \
    --output leaks.csv --output-clean train_clean.csv

# 4) Summary: cluster-size distribution.
ppisplit summary \
    --cluster-dir splits/run1 \
    --csv train.csv \
    --test-csv test.csv --top 5
```

`cluster` writes the hit table and the similarity rule to its output directory, and the later commands read both from `--cluster-dir`. `split` assigns whole clusters to folds, either greedy balanced k-fold (`--mode kfold --n-folds N`) or repeated random splits (`--mode montecarlo --n-splits N`). With `--test-csv` it drops the test complexes from the pool but keeps training complexes that share a cluster with them; `--exclude-test-collateral` drops those too. `clean-test-set` writes a test CSV without the complexes that leak into any of the given training sets.

## Python API

`hits.parquet` holds every alignment the search found, so thresholds can be compared without running MMseqs2 again:

```python
import pandas as pd

from ppisplit.predicates import COV_MODE_SYMMETRIC, HyperbolaPredicate

hits = pd.read_parquet("out/example/hits.parquet")
for x in [0.16, 0.24, 0.32]:
    pred = HyperbolaPredicate(x=x, cov_mode=COV_MODE_SYMMETRIC)
    print(x, int(pred(hits).sum()))
```

## Design decisions

**Sequences come from the coordinates.** With `--sequence-source resolved`, the default, a chain's sequence is the amino-acid residues present in model 0 of the structure, and a chain with none (DNA, a ligand) is left out of the comparison. SEQRES sequences also contain purification tags, signal peptides and unresolved loops, and that causes errors in both directions. In a set of 8,515 chains, 1,290 carry a purification-tag motif, and 94.7% of those tags are unresolved. Under SEQRES, `9f6o` scores 0.909 against the anti-myc Fab peptide `2or9_P` because of the tag alone; from the coordinates it scores 0.044. Unresolved residues also enlarge the coverage denominator and hide real similarity: `6lsa_F ~ 4myw_A` scores 0.208 under SEQRES and 0.263 from the coordinates, on the same alignment. `--sequence-source seqres` exists to reproduce older results.

**Search settings never filter a hit.** MMseqs2 runs with no identity or coverage floor, sensitivity 7.5, E-value `1e30`, and `--max-seqs` set from the size of the target database, so every hit reaches the similarity rule. A lower E-value cutoff makes verdicts depend on the database size: with `-e 0.1`, the same chain pairs got 73,419 different verdicts in a 4,574-chain and an 8,515-chain database, and with `1e30` none did. The settings are in `ppisplit.search.SEARCH_SETTINGS`.

**Complexes are joined by direct chain similarity.** Two complexes are adjacent only if their chains are similar to each other, and the clusters are the connected components of that complex graph. Joining complexes whose chains merely fall in the same chain cluster would merge unrelated complexes through chains of intermediates: in one 2,474-chain cluster, 76.6% of the chain pairs it joined were not similar to each other. Chain clusters (`chain_clusters_from_edges`) are still computed, for diagnostics only.

## Modules

| module | contents |
|---|---|
| `ppisplit.io` | annotation loading and PDB/mmCIF to FASTA |
| `ppisplit.search` | the MMseqs2 `easy-search` wrapper and its settings |
| `ppisplit.predicates` | similarity rules: `HyperbolaPredicate`, `RectanglePredicate`, `UnionOfPredicates` |
| `ppisplit.graph` | chain clusters, complex clusters, removal of test clusters from the training pool |
| `ppisplit.folds` | k-fold and Monte Carlo fold assignment |
| `ppisplit.leaks` | `detect_leaks`, the test-against-train leak report |

## Tests

```bash
uv sync --extra test
uv run pytest
```

The end-to-end search test is skipped when `mmseqs` is not on `PATH`.

## Status

ppisplit is pre-1.0 and its API may change. Pin a commit or tag if you depend on it.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT, see [`LICENSE`](LICENSE).
