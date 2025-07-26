import os
import argparse
import numpy as np
import pandas as pd
from Bio import SeqIO

from consts import RND_STATE, DEFAULT_LABEL_COLUMN


def parse_args():
    parser = argparse.ArgumentParser(description="Sample sequences from serogroups.")
    parser.add_argument("--fetch_fasta", action="store_true", help="Fetch sequences in FASTA format, otherwise use embeddings.")
    parser.add_argument("--sequences", type=str, help="Path to the sequences file in FASTA format.")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to the embeddings file in NPY/NPZ format.")
    parser.add_argument("--labels", type=str, required=True, help="Path to the labels file in CSV format.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save sampled sequences.")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of sequences to sample from each serogroup.")
    parser.add_argument("--sample_all", action="store_true", help="Fetch all sequences if set, ignoring sample size.")
    parser.add_argument("--serogroups", type=str, default=None, help="Comma-separated list of serogroups to sample from. If not provided, all serogroups will be sampled.")
    
    args = parser.parse_args()
    if args.fetch_fasta and args.sequences is None:
        raise ValueError("If fetch_fasta is set, sequences must be provided.")
    if args.sample_size is not None and args.sample_all:
        raise ValueError("Cannot specify both sample_size and sample_all. Choose one.")
    if args.sample_size is None and not args.sample_all:
        raise ValueError("Must specify either sample_size or set sample_all to True.")
    if args.serogroups:
        args.serogroups = [sg.strip() for sg in args.serogroups.split(",")]
    return args


def main(args):
    ### TODO move to params
    serotype_column = DEFAULT_LABEL_COLUMN
    ###
    print("Loading data...")
    X = np.load(args.embeddings)  # shape (N, D)
    is_X_npz = isinstance(X, np.lib.npyio.NpzFile)
    labels = pd.read_csv(args.labels, sep="\t", index_col=0)
    if not is_X_npz:
        assert X.shape[0] == len(labels), "Number of embeddings and labels do not match."

    # Get unique serogroups
    serogroups = labels[serotype_column].unique()
    if args.serogroups:
        serogroups = [sg for sg in serogroups if sg in args.serogroups]
        print(f"Filtering serogroups: {args.serogroups}")
    else:
        print("Sampling from all serogroups.")

    id_mapping = {}
    if args.fetch_fasta:
        print(f"Reading sequences from {args.sequences}...")
        for record in SeqIO.parse(args.sequences, "fasta"):
            public_id = record.id.split("__")[0]
            id_mapping[public_id] = record.seq
    # Sample sequences
    sampled_sequences = []
    for sg in serogroups:
        sg_indices = labels[labels['Serotype'] == sg].index
        if args.sample_all:
            sampled_indices = sg_indices.tolist()
        else:
            sampled_indices = sg_indices.sample(n=args.sample_size, random_state=RND_STATE).tolist()
        
        for idx in sampled_indices:
            if args.fetch_fasta:
                record = id_mapping.get(idx, None)
                if record is None:
                    print(f"Warning: Sequence for ID {idx} not found in the FASTA file.")
                    continue
                sampled_sequences.append(record)
            else:
                if is_X_npz:
                    embedding = X[labels["Public_Name"].get_loc(idx)].tolist()
                else:
                    embedding = X[labels.index.get_loc(idx)].tolist()
                sampled_sequences.append({
                    "ID": idx,
                    "Serotype": sg,
                    "Embedding": embedding
                })
    print(f"Sampled {len(sampled_sequences)} sequences from {len(serogroups)} serogroup(s).")

    # Save sampled sequences
    outpref = f"{args.output_dir}/sequences" + (f"_{RND_STATE}_" + "_".join(args.serogroups) if args.serogroups else "all")
    if args.fetch_fasta:
        SeqIO.write(sampled_sequences, outpref + ".fasta", "fasta")
    else:
        pd.DataFrame(sampled_sequences).to_csv(outpref + ".csv", index=False)


if __name__ == "__main__":
    main(parse_args())