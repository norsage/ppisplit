# Changelog

## 0.1.3

- Cluster ids no longer depend on `PYTHONHASHSEED`. Components were numbered by their union-find root, and the root followed the iteration order of a set of uids, which changes from one Python process to the next. `ppisplit split` on the same cluster directory and seed could therefore assign different clusters to val. Clusters are now numbered by their smallest member. Folds written by 0.1.2 and earlier are not reproducible; rebuild them with 0.1.3.

## 0.1.1

- New sequence source `seqres-trimmed`: each chain's SEQRES from the first to the last resolved residue. It is now the default for `ppisplit cluster`. Unresolved termini and purification tags stay out; unresolved internal loops are kept, so a verdict no longer depends on whether a crystal ordered a loop.
- Hit tables built with 0.1.0 used `resolved`; pass `--sequence-source resolved` to reproduce them.

## 0.1.0

- Initial release.
