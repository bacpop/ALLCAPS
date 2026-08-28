"""Split paired FASTA and metadata files into train/test sets.

Pools all sources, applies unified label preprocessing, then performs a
sample-grouped, stratified train/test split with per-source ratios.

The split unit is the *sample* (``--id_column``, e.g. Public_ID), not the
contig: every contig of a sample is kept on the same side, so sibling contigs
of one assembly never straddle the train/test boundary (no leakage). Serotypes
with too few distinct *samples* to appear on both sides -- globally rarer than
``MIN_SEROTYPE_COUNT`` samples, or too rare within a source -- are placed
entirely in training, ensuring every label in the test set also appears in
training. A runtime assertion guards the no-leakage invariant.
"""

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split

from ..consts import (
    CONTIG_SEP,
    DEFAULT_LABEL_COLUMN,
    MIN_SEROTYPE_COUNT,
    RND_STATE,
    TRAIN_SPLIT_RATIO,
    DEFAULT_ID_COLUMN,
    DEFAULT_CONTIG_COLUMN,
)
from ..data_labels_preprocessing import preprocess_metadata
from ..logging_config import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/test split for FASTA + metadata pairs"
    )
    parser.add_argument(
        "--fastas",
        nargs="+",
        required=True,
        help="List of input FASTA files (space-separated)",
    )
    parser.add_argument(
        "--metadata",
        nargs="+",
        required=True,
        help="List of metadata CSV/TSV files aligned to FASTAs (space-separated)",
    )
    parser.add_argument(
        "--ratios",
        type=str,
        default=str(TRAIN_SPLIT_RATIO),
        help="Train ratios: single float or comma-separated list matching number of inputs",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write train/test FASTA and metadata",
    )
    parser.add_argument(
        "--seed", type=int, default=RND_STATE, help="Random seed for splits"
    )
    parser.add_argument(
        "--id_column",
        type=str,
        default=DEFAULT_ID_COLUMN,
        help="Metadata column that matches FASTA record.id",
    )
    parser.add_argument(
        "--serotype_column",
        type=str,
        default=DEFAULT_LABEL_COLUMN,
        help="Metadata column containing class labels for stratification",
    )

    args = parser.parse_args()
    assert len(args.fastas) == len(args.metadata), (
        f"fastas and metadata must have same length (got {len(args.fastas)} vs {len(args.metadata)})"
    )
    return args


def parse_ratios(ratios_raw: str, n: int) -> List[float]:
    if "," in ratios_raw:
        ratios = [float(r.strip()) for r in ratios_raw.split(",") if r.strip()]
        assert len(ratios) == n, (
            f"Provided {len(ratios)} ratios but {n} inputs; counts must match."
        )
    else:
        ratio = float(ratios_raw)
        ratios = [ratio] * n
    assert all(0.0 < r < 1.0 for r in ratios), "Train ratios must be in (0,1)"
    return ratios


def load_and_align(
    fasta_path, meta_path, id_column, contig_column=DEFAULT_CONTIG_COLUMN
):
    """Load a FASTA file and its metadata, align rows by composite ID.

    Returns
    -------
    meta : pd.DataFrame   – aligned metadata (reset index)
    records : list[SeqRecord] – FASTA records in matching order
    """
    meta = pd.read_csv(meta_path, sep="\t" if meta_path.endswith(".tsv") else ",")
    fasta_records = list(SeqIO.parse(fasta_path, "fasta"))

    meta_ids = (
        meta[id_column].astype(str) + CONTIG_SEP + meta[contig_column].astype(str)
    )
    fasta_id_set = set(rec.id for rec in fasta_records)

    keep_mask = meta_ids.isin(fasta_id_set)
    if not keep_mask.all():
        dropped = (~keep_mask).sum()
        logger.warning("Dropping %d metadata rows not found in %s", dropped, fasta_path)
    meta = meta[keep_mask].copy()
    meta_ids = meta_ids[keep_mask]

    id_to_rec = {rec.id: rec for rec in fasta_records}
    records = [id_to_rec[mid] for mid in meta_ids if mid in id_to_rec]

    return meta.reset_index(drop=True), records


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    ratios = parse_ratios(args.ratios, len(args.fastas))
    os.makedirs(args.output_dir, exist_ok=True)

    serotype_col = args.serotype_column
    id_col = args.id_column  # sample-level grouping key (e.g. Public_ID)

    # ── Phase 1: Load and align every source ─────────────────────────
    all_records: list = []
    meta_parts: list[pd.DataFrame] = []
    for i, (fasta_path, meta_path) in enumerate(zip(args.fastas, args.metadata)):
        logger.info("Loading source %d: %s", i, fasta_path)
        meta, records = load_and_align(fasta_path, meta_path, args.id_column)
        meta["_source"] = i
        meta["_rec_offset"] = range(len(all_records), len(all_records) + len(records))
        meta_parts.append(meta)
        all_records.extend(records)

    combined = pd.concat(meta_parts, ignore_index=True)
    logger.info(
        "Combined pool: %d contigs, %d samples, %d serotypes",
        len(combined),
        combined[id_col].nunique(),
        combined[serotype_col].nunique(),
    )

    # ── Phase 2: Unified label preprocessing ─────────────────────────
    n_before = len(combined)
    combined = preprocess_metadata(combined, serotype_column=serotype_col)
    logger.info(
        "After preprocessing: %d contigs (dropped %d), %d serotypes",
        len(combined),
        n_before - len(combined),
        combined[serotype_col].nunique(),
    )

    # ── Phase 3: Global rare-serotype detection (counted in samples) ──
    # Rarity is measured in distinct samples (Public_ID), not contigs, so a
    # serotype seen across many contigs of only a handful of samples is still
    # treated as rare and kept entirely in train.
    global_counts = combined.groupby(serotype_col)[id_col].nunique()
    rare_global = set(global_counts[global_counts < MIN_SEROTYPE_COUNT].index)
    if rare_global:
        n_rare = combined[serotype_col].isin(rare_global).sum()
        logger.info(
            "Globally rare serotypes (all -> train): %s (%d contigs)",
            sorted(rare_global),
            n_rare,
        )

    rare_mask = combined[serotype_col].isin(rare_global)
    train_idx: list[int] = combined.index[rare_mask].tolist()
    test_idx: list[int] = []

    # ── Phase 4: Per-source, sample-grouped stratified split ─────────
    # The split unit is the *sample* (id_col, e.g. Public_ID), never the
    # contig: every contig of a sample goes to the same side, which eliminates
    # train/test leakage. Stratification and rare-class gating are therefore
    # also expressed in sample units.
    non_rare = combined[~rare_mask]
    for src_i, ratio in enumerate(ratios):
        src = non_rare[non_rare["_source"] == src_i]
        if len(src) == 0:
            continue

        # Collapse contigs to one row per sample carrying its serotype. One
        # serotype per sample is guaranteed upstream (the drop_duplicates
        # merge in build_contig_metadata); .first() is a defensive tie-break.
        group_serotype = src.groupby(id_col)[serotype_col].first()
        group_ids = group_serotype.index.to_numpy()
        group_labels = group_serotype.to_numpy()
        logger.info(
            "Source %d: %d samples across %d contigs, ratio=%s",
            src_i,
            len(group_ids),
            len(src),
            ratio,
        )

        # Serotypes with too few *samples* in this source to place one on each
        # side of the split -> keep all their contigs in train. The 1e-9 guards
        # against float error (e.g. 1 - 0.8 = 0.1999.. would inflate the ceil).
        test_ratio = 1 - ratio
        min_per_class = max(int(np.ceil(1.0 / test_ratio - 1e-9)), 2)
        src_counts = pd.Series(group_labels).value_counts()
        src_rare = set(src_counts[src_counts < min_per_class].index)
        if src_rare:
            logger.info("Source-rare (-> train): %s", sorted(src_rare))
            rare_group_mask = np.isin(group_labels, list(src_rare))
            rare_groups = set(group_ids[rare_group_mask])
            train_idx.extend(src.index[src[id_col].isin(rare_groups)].tolist())
            group_ids = group_ids[~rare_group_mask]
            group_labels = group_labels[~rare_group_mask]

        if len(group_ids) == 0:
            continue

        tr_g, te_g = train_test_split(
            group_ids,
            test_size=test_ratio,
            stratify=group_labels,
            random_state=rng.integers(0, 2**32 - 1),
        )
        tr_g, te_g = set(tr_g), set(te_g)
        train_idx.extend(src.index[src[id_col].isin(tr_g)].tolist())
        test_idx.extend(src.index[src[id_col].isin(te_g)].tolist())

    # ── Phase 5: Verify no sample leakage and label consistency ──────
    train_meta = combined.loc[train_idx]
    test_meta = combined.loc[test_idx]

    # Hard guard: no sample (Public_ID) may appear on both sides.
    overlap = set(train_meta[id_col]) & set(test_meta[id_col])
    assert not overlap, (
        f"Sample leakage: {len(overlap)} {id_col}(s) in both train and test, "
        f"e.g. {sorted(overlap)[:10]}"
    )
    logger.info(
        "No sample leakage: %d train samples / %d test samples, disjoint.",
        train_meta[id_col].nunique(),
        test_meta[id_col].nunique(),
    )

    train_labels = set(train_meta[serotype_col])
    test_labels = set(test_meta[serotype_col])
    test_only = test_labels - train_labels
    if test_only:
        logger.warning("Serotypes in test but NOT in train: %s", sorted(test_only))
    else:
        logger.info("All %d test serotypes are present in train.", len(test_labels))

    logger.info(
        "Train: %d contigs | Test: %d contigs", len(train_meta), len(test_meta)
    )

    # ── Phase 6: Write outputs ───────────────────────────────────────
    train_records = [all_records[i] for i in train_meta["_rec_offset"]]
    test_records = [all_records[i] for i in test_meta["_rec_offset"]]

    # Drop internal bookkeeping columns
    out_cols = [c for c in combined.columns if not c.startswith("_")]

    train_fasta = os.path.join(args.output_dir, "train.fasta")
    test_fasta = os.path.join(args.output_dir, "test.fasta")
    with open(train_fasta, "w") as fh:
        SeqIO.write(train_records, fh, "fasta")
    with open(test_fasta, "w") as fh:
        SeqIO.write(test_records, fh, "fasta")

    train_meta[out_cols].to_csv(
        os.path.join(args.output_dir, "train_metadata.csv"), index=False
    )
    test_meta[out_cols].to_csv(
        os.path.join(args.output_dir, "test_metadata.csv"), index=False
    )

    logger.info("Wrote train/test files to %s", args.output_dir)


if __name__ == "__main__":
    main()
