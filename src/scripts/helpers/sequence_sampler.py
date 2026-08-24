import os
import argparse
import numpy as np
import pandas as pd
from Bio import SeqIO

from ..consts import RND_STATE, DEFAULT_LABEL_COLUMN
from ..logging_config import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Sample sequences from serogroups.")
    parser.add_argument(
        "--fetch_fasta",
        action="store_true",
        help="Fetch sequences in FASTA format, otherwise use embeddings.",
    )
    parser.add_argument(
        "--sequences", type=str, help="Path to the sequences file in FASTA format."
    )
    parser.add_argument(
        "--embeddings", type=str, help="Path to the embeddings file in NPY/NPZ format."
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to the labels file in CSV format.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save sampled sequences.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of sequences to sample from each serogroup.",
    )
    parser.add_argument(
        "--sample_all",
        action="store_true",
        help="Fetch all sequences if set, ignoring sample size.",
    )
    parser.add_argument(
        "--serogroups",
        type=str,
        default=None,
        help="Comma-separated list of serogroups to sample from. If not provided, all serogroups will be sampled.",
    )
    parser.add_argument(
        "--id_column",
        type=str,
        default="ERR",
        help="Column name for the ID field in the labels file.",
    )

    args = parser.parse_args()
    if not (args.fetch_fasta or args.embeddings):
        raise parser.error("Either --fetch_fasta or --embeddings must be provided.")
    if args.fetch_fasta and args.sequences is None:
        raise parser.error("If fetch_fasta is set, sequences must be provided.")
    if args.sample_size is not None and args.sample_all:
        raise parser.error(
            "Cannot specify both sample_size and sample_all. Choose one."
        )
    if args.sample_size is None and not args.sample_all:
        raise parser.error("Must specify either sample_size or set sample_all to True.")
    if args.serogroups:
        args.serogroups = [sg.strip() for sg in args.serogroups.split(",")]
    return args


def main(args):
    serotype_column = DEFAULT_LABEL_COLUMN
    id_column = args.id_column

    logger.info("Loading data...")
    labels = (
        pd.read_csv(args.labels, sep="\t" if args.labels.endswith(".tsv") else ",")
        .set_index(id_column)
        .drop_duplicates()
    )  # TODO clean up labels
    if args.embeddings:
        X = np.load(args.embeddings)  # shape (N, D)
        is_X_npz = isinstance(X, np.lib.npyio.NpzFile)
        if not is_X_npz:
            assert X.shape[0] == len(labels), (
                "Number of embeddings and labels do not match."
            )

    serogroups = labels[serotype_column].unique()
    if args.serogroups:
        serogroups = [sg for sg in serogroups if sg in args.serogroups]
        logger.info("Filtering serogroups: %s", args.serogroups)
    else:
        logger.info("Sampling from all serogroups.")

    id_mapping = {}
    if args.fetch_fasta:
        logger.info("Reading sequences from %s...", args.sequences)
        for record in SeqIO.parse(args.sequences, "fasta"):
            # Record ids are `Public_ID#Contig_ID` (non-CBL keeps its `NONCBL#`
            # prefix, so strip only the trailing contig field). A sample's cps
            # locus can span several contigs — keep every one of them, otherwise
            # later contigs overwrite earlier ones and the query set silently
            # collapses to one arbitrary contig per sample.
            public_id = record.id.rsplit("#", 1)[0]
            id_mapping.setdefault(public_id, []).append(record)

    sampled_sequences = []
    sampled_ids = []  # Track sampled IDs for labels subset
    if len(serogroups) == 1 or args.sample_all:
        # Single serogroup or sampling all: use original logic
        for sg in serogroups:
            # One entry per sample: `labels` is indexed by Public_ID and holds one
            # row per contig, so a multi-contig sample would otherwise be visited
            # (and emitted) once per contig.
            sg_indices = labels[labels[serotype_column] == sg].index.unique()
            if args.sample_all:
                sampled_indices = sg_indices.tolist()
            else:
                sample_size_sg = min(args.sample_size, len(sg_indices))
                sampled_indices = (
                    np.random.RandomState(RND_STATE)
                    .choice(sg_indices, size=sample_size_sg, replace=False)
                    .tolist()
                )

            for idx in sampled_indices:
                sampled_ids.append(idx)  # Track sampled ID
                if args.fetch_fasta:
                    fasta_key = f"{idx}"
                    records = id_mapping.get(fasta_key)
                    if not records:
                        logger.warning(
                            "Sequence for ID %s not found in the FASTA file.", fasta_key
                        )
                        continue
                    sampled_sequences.extend(records)
                else:
                    if is_X_npz:
                        # For NPZ files, the key should match the sample ID format
                        key_candidates = [idx, f"cbl|{idx}", f"non-cbl|{idx}"]
                        embedding = None
                        for key in key_candidates:
                            if key in X:
                                embedding = X[key].tolist()
                                break
                        if embedding is None:
                            logger.warning(
                                "Embedding for ID %s not found in NPZ file.", idx
                            )
                            continue
                    else:
                        embedding = X[labels.index.get_loc(idx)].tolist()

                    sampled_sequences.append(
                        {"ID": idx, "Serotype": sg, "Embedding": embedding}
                    )
    else:
        # Multiple serogroups: weighted sampling to maintain distribution
        if not args.sample_size:
            raise ValueError(
                "sample_size must be specified for weighted sampling across multiple serogroups"
            )

        # Counted in samples (unique Public_ID), not contig rows, so the weights
        # and the per-serogroup cap agree with the deduplicated sg_indices below.
        serogroup_counts = {
            sg: labels[labels[serotype_column] == sg].index.nunique()
            for sg in serogroups
        }
        total_samples = sum(serogroup_counts.values())
        serogroup_sample_sizes = {}

        for sg in serogroups:
            proportion = serogroup_counts[sg] / total_samples
            sample_size_sg = max(
                1, round(args.sample_size * proportion)
            )  # Ensure at least 1 sample
            sample_size_sg = min(
                sample_size_sg, serogroup_counts[sg]
            )  # Don't exceed available samples
            serogroup_sample_sizes[sg] = sample_size_sg
        # TODO Adjust if we're over/under the target due to rounding for precision

        logger.info("Weighted sampling plan (target: %d total):", args.sample_size)
        for sg, sample_size_sg in serogroup_sample_sizes.items():
            proportion = sample_size_sg / sum(serogroup_sample_sizes.values())
            logger.info(
                "\t%s: %d samples (%.2f%%)", sg, sample_size_sg, 100.0 * proportion
            )

        for sg in serogroups:
            sg_indices = labels[labels[serotype_column] == sg].index.unique()
            sample_size_sg = serogroup_sample_sizes[sg]

            if sample_size_sg > 0:
                sampled_indices = (
                    np.random.RandomState(RND_STATE)
                    .choice(sg_indices, size=sample_size_sg, replace=False)
                    .tolist()
                )

                for idx in sampled_indices:
                    sampled_ids.append(idx)  # Track sampled ID
                    if args.fetch_fasta:
                        fasta_key = f"{idx}"
                        records = id_mapping.get(fasta_key)
                        if not records:
                            logger.warning(
                                "Sequence for ID %s not found in the FASTA file.",
                                fasta_key,
                            )
                            continue
                        sampled_sequences.extend(records)
                    else:
                        if is_X_npz:
                            # For NPZ files, the key should match the sample ID format
                            key_candidates = [idx, f"cbl|{idx}", f"non-cbl|{idx}"]
                            embedding = None
                            for key in key_candidates:
                                if key in X:
                                    embedding = X[key].tolist()
                                    break
                            if embedding is None:
                                logger.warning(
                                    "Embedding for ID %s not found in NPZ file.", idx
                                )
                                continue
                        else:
                            embedding = X[labels.index.get_loc(idx)].tolist()

                        sampled_sequences.append(
                            {"ID": idx, "Serotype": sg, "Embedding": embedding}
                        )
    assert sampled_sequences, (
        "No sequences were sampled. Please check your input data and sampling parameters."
    )
    logger.info(
        "Sampled %d sequences from %d serogroup(s).",
        len(sampled_sequences),
        len(serogroups),
    )

    # Save sampled sequences
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    serogroups_str = "_".join(list(map(lambda s: s.replace("/", ""), args.serogroups)))
    if len(serogroups) > 1 and not args.sample_all:
        outpref = f"{args.output_dir}/sequences_weighted_{args.sample_size}_{RND_STATE}"
        if args.serogroups:
            outpref += "_" + serogroups_str
    else:
        outpref = f"{args.output_dir}/sequences" + (
            f"_{RND_STATE}_" + serogroups_str if args.serogroups else "_all"
        )

    if args.fetch_fasta:
        SeqIO.write(sampled_sequences, outpref + ".fasta", "fasta")
        logger.info("Saved FASTA sequences to: %s.fasta", outpref)
    else:
        pd.DataFrame(sampled_sequences).to_csv(outpref + ".csv", index=False)
        logger.info("Saved embeddings to: %s.csv", outpref)

    # Save subset of labels corresponding to sampled sequences
    sampled_labels = labels.loc[sampled_ids].copy()
    sampled_labels.to_csv(outpref + "_labels.tsv", sep="\t")
    logger.info(
        "Saved %d corresponding labels to: %s_labels.tsv", len(sampled_labels), outpref
    )


if __name__ == "__main__":
    main(parse_args())
