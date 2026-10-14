# Changelog

## 0.1.1

- New sequence source `seqres-trimmed`: each chain's SEQRES from the first to the last resolved residue. It is now the default for `ppisplit cluster`. Unresolved termini and purification tags stay out; unresolved internal loops are kept, so a verdict no longer depends on whether a crystal ordered a loop.
- Hit tables built with 0.1.0 used `resolved`; pass `--sequence-source resolved` to reproduce them.

## 0.1.0

- Initial release.
