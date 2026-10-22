# Changelog

## 0.2.0

- New split mode `montecarlo-balanced`. In `montecarlo` the val share of a split depends on which cluster is drawn last: one large cluster can push it from 25% to 35%. The new mode sends each cluster above `--large-threshold` to val with probability `--val-ratio` and fills the rest with small clusters, skipping those that would overshoot, so val is `--val-ratio` of the total up to one small cluster while every cluster still lands in val with probability close to `--val-ratio`. The mode fails if the large clusters alone exceed `--val-ratio`.
- `split --weights CSV` (columns uid, weight; `montecarlo-balanced` only): cluster sizes and the val share are counted in weights instead of complexes.
- Existing modes are unchanged: their folds are the same as with 0.1.4.

## 0.1.4

- Uids keep their case. `write_fasta` lower-cased uids while chain keys did not, so for an upper-case uid every command reading the cluster directory found no chains: `split` wrote an empty fold file (or raised `KeyError` in kfold mode), `check-leaks` lost every sequence leak and `summary` printed zeros, all with exit 0. FASTA headers now carry the uid as given.
- `split`, `summary`, `check-leaks` and `clean-test-set` exit 1 when no complex of the CSV has a chain in the cluster directory's FASTA, instead of reporting empty results.
- The purpose-split search (`cluster --target-csv`) no longer drops hits whose query and target share an identifier. Such a hit is one chain present in both databases, not a self-match; the direct uid-overlap check already put these complexes in the verdict, but the similarity rule never saw their hits. Verdicts are unchanged on the sets re-audited.
- Hit tables written by 0.1.3 and earlier lack the same-identifier hits; rebuild them to have those hits in the table. Folds and leak reports built from upper-case uids were empty and must be rebuilt.

## 0.1.3

- Cluster ids no longer depend on `PYTHONHASHSEED`. Components were numbered by their union-find root, and the root followed the iteration order of a set of uids, which changes from one Python process to the next. `ppisplit split` on the same cluster directory and seed could therefore assign different clusters to val. Clusters are now numbered by their smallest member. Folds written by 0.1.2 and earlier are not reproducible; rebuild them with 0.1.3.

## 0.1.1

- New sequence source `seqres-trimmed`: each chain's SEQRES from the first to the last resolved residue. It is now the default for `ppisplit cluster`. Unresolved termini and purification tags stay out; unresolved internal loops are kept, so a verdict no longer depends on whether a crystal ordered a loop.
- Hit tables built with 0.1.0 used `resolved`; pass `--sequence-source resolved` to reproduce them.

## 0.1.0

- Initial release.
