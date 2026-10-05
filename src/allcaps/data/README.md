# Packaged data

`dexB_aliA_ATCC700669.fasta` — the `dexB` and `aliA` flanking genes of
*S. pneumoniae* ATCC700669 (FM211187.1), used by `allcaps.data_locus_cutter` to cut the
*cps* locus out of an assembly. Two records sharing one id, in positive-strand order
(`dexB` → `aliA`); the cutter asserts exactly two sequences per id.

Shipped inside the package so `ALLCAPS predict --extract align` works from any directory.
Resolve it with `allcaps.cli.artifacts.packaged_flanks()`, never a relative path.
