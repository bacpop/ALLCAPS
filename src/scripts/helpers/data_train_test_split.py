"""Split paired FASTA and metadata files into train/test sets.

Takes lists of FASTA paths and matching metadata CSV paths (one-to-one).
Applies either a single train ratio to all, or per-file ratios via a
comma-separated list. Outputs combined train/test FASTA and metadata
CSV files in the specified output directory.
"""

import argparse
import os
from typing import List, Sequence

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split

from ..consts import DEFAULT_LABEL_COLUMN, RND_STATE, TRAIN_SPLIT_RATIO, CONTIG_SEP
from ..data_labels_preprocessing import cleanup_serotype
DEFAULT_ID_COLUMN = "Public_ID"
DEFAULT_CONTIG_COLUMN = "Contig_ID"


def parse_args():
    parser = argparse.ArgumentParser(description="Train/test split for FASTA + metadata pairs")
    parser.add_argument("--fastas", nargs="+", required=True,
                        help="List of input FASTA files (space-separated)")
    parser.add_argument("--metadata", nargs="+", required=True,
                        help="List of metadata CSV/TSV files aligned to FASTAs (space-separated)")
    parser.add_argument("--ratios", type=str, default=str(TRAIN_SPLIT_RATIO),
                        help="Train ratios: single float or comma-separated list matching number of inputs")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write train/test FASTA and metadata")
    parser.add_argument("--seed", type=int, default=RND_STATE, help="Random seed for splits")
    parser.add_argument("--id_column", type=str, default=DEFAULT_ID_COLUMN,
                        help="Metadata column that matches FASTA record.id")
    parser.add_argument("--serotype_column", type=str, default=DEFAULT_LABEL_COLUMN,
                        help="Metadata column containing class labels for stratification")

    args = parser.parse_args()
    assert len(args.fastas) == len(args.metadata), \
        f"fastas and metadata must have same length (got {len(args.fastas)} vs {len(args.metadata)})"
    return args


def parse_ratios(ratios_raw: str, n: int) -> List[float]:
    if "," in ratios_raw:
        ratios = [float(r.strip()) for r in ratios_raw.split(",") if r.strip()]
        assert len(ratios) == n, f"Provided {len(ratios)} ratios but {n} inputs; counts must match."
    else:
        ratio = float(ratios_raw)
        ratios = [ratio] * n
    assert all([0.0 < r < 1.0 for r in ratios]), "Train ratios must be in (0,1)"
    return ratios


def split_one(
    fasta_path: str,
    meta_path: str,
    train_ratio: float,
    id_column: str,
    serotype_column: str,
    rng: np.random.Generator,
):
    contig_column = DEFAULT_CONTIG_COLUMN

    # Load data
    meta = pd.read_csv(meta_path, sep="\t" if meta_path.endswith(".tsv") else ",")
    fasta_records = list(SeqIO.parse(fasta_path, "fasta"))

    # Determine IDs
    meta_ids = meta[id_column].astype(str) + CONTIG_SEP + meta[contig_column].astype(str)
    fasta_id_set = set(rec.id for rec in fasta_records)

    # Filter metadata to those present in FASTA
    keep_mask = meta_ids.isin(fasta_id_set)
    if not keep_mask.all():
        dropped = (~keep_mask).sum()
        print(f"Warning: dropping {dropped} metadata rows not found in FASTA {fasta_path}")
    meta = meta[keep_mask].copy()
    meta_ids = meta_ids[keep_mask]

    # Align records
    id_to_rec = {rec.id: rec for rec in fasta_records}
    records = [id_to_rec[i] for i in meta_ids if i in id_to_rec]

    # Stratified split on serotype; handle rare classes
    serotypes = meta[serotype_column] if serotype_column in meta.columns else None
    if serotypes is not None:
        # Apply the same label-cleaning used during training so that
        # train/test splits are based on consistent, canonical labels.
        serotypes = serotypes.map(cleanup_serotype)
        meta[serotype_column] = serotypes
    idx_all = np.arange(len(records))

    if serotypes is None:
        print(f"Warning: serotype column '{serotype_column}' not found in {meta_path}; falling back to random split.")
        test_size = 1 - train_ratio
        train_idx, test_idx = train_test_split(
            idx_all, test_size=test_size, random_state=rng.integers(0, 2**32 - 1)
        )
    else:
        counts = serotypes.value_counts()
        rail_classes = counts[(counts * train_ratio < 1) | (counts * (1 - train_ratio) < 1)].index.tolist()
        if rail_classes:
            print(f"Classes too small for stratified split (sent to train): {[(c, int(counts[c])) for c in rail_classes]}")
        rail_mask = serotypes.isin(rail_classes)
        idx_rail = idx_all[rail_mask.to_numpy()]
        idx_rest = idx_all[~rail_mask.to_numpy()]
        serotypes_rest = serotypes[~rail_mask]

        if len(np.unique(serotypes_rest)) < 2 or len(idx_rest) == 0:
            train_idx = idx_all
            test_idx = np.array([], dtype=int)
        else:
            test_size = 1 - train_ratio
            train_idx_rest, test_idx_rest = train_test_split(
                idx_rest,
                test_size=test_size,
                stratify=serotypes_rest,
                random_state=rng.integers(0, 2**32 - 1),
            )
            train_idx = np.concatenate([idx_rail, train_idx_rest])
            test_idx = test_idx_rest

    train_records = [records[i] for i in train_idx]
    test_records = [records[i] for i in test_idx]
    train_meta = meta.iloc[train_idx]
    test_meta = meta.iloc[test_idx]

    return train_records, test_records, train_meta, test_meta


def main():
    args = parse_args()
    fastas: Sequence[str] = args.fastas
    metas: Sequence[str] = args.metadata

    rng = np.random.default_rng(args.seed)
    ratios = parse_ratios(args.ratios, len(fastas))
    os.makedirs(args.output_dir, exist_ok=True)

    train_fasta_path = os.path.join(args.output_dir, "train.fasta")
    test_fasta_path = os.path.join(args.output_dir, "test.fasta")
    train_meta_path = os.path.join(args.output_dir, "train_metadata.csv")
    test_meta_path = os.path.join(args.output_dir, "test_metadata.csv")

    all_train_meta, all_test_meta = [], []
    for fasta_path, meta_path, ratio in zip(fastas, metas, ratios):
        print(f"\nSplitting {fasta_path} with {meta_path} at train_ratio={ratio}")
        train_recs, test_recs, train_meta, test_meta = split_one(
            fasta_path, meta_path, ratio, args.id_column, args.serotype_column, rng
        )

        # Append FASTA records
        with open(train_fasta_path, "a") as f_train:
            SeqIO.write(train_recs, f_train, "fasta")
        with open(test_fasta_path, "a") as f_test:
            SeqIO.write(test_recs, f_test, "fasta")

        all_train_meta.append(train_meta)
        all_test_meta.append(test_meta)

    # Concatenate metadata and save
    if all_train_meta:
        pd.concat(all_train_meta, ignore_index=True).to_csv(train_meta_path, index=False)
    if all_test_meta:
        pd.concat(all_test_meta, ignore_index=True).to_csv(test_meta_path, index=False)

    print(f"\nWrote train/test fasta/metadata files to {args.output_dir}.")


if __name__ == "__main__":
    main()
