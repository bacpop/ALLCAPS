import argparse
from Bio import SeqIO
import numpy as np
import pandas as pd

from consts import DEFAULT_LABEL_COLUMN, DEFAULT_MISSING_LABEL

def main(args):
    ### TODO move to params
    label_column = DEFAULT_LABEL_COLUMN
    missing_label = DEFAULT_MISSING_LABEL
    ###
    labels = pd.read_csv(args.labels, index_col=0, sep="\t")

    is_duplicate = labels.duplicated()
    if is_duplicate.any():
        print("Dropping duplicate label rows...")
        labels = labels[~is_duplicate]
    labels[label_column] = labels[label_column].fillna(missing_label)
    indices = labels[label_column] != missing_label
    labels = labels[indices][label_column]

    if args.store_only is not None:
        labels = labels[labels.isin(args.store_only)]
        print(f"Storing only serotypes: {set(args.store_only) & set(labels)}")
    else:
        print("Storing all serotypes.")

    sequences_dict = {record.id.split("__")[0]: record.seq for record in SeqIO.parse(args.fasta, "fasta")}
    
    print(f"Found {len(sequences_dict)} sequences in the fasta file.")
    if not sequences_dict:
        print("No relevant sequences found. Exiting.")
        return
    
    results = []
    for public_id in labels.index:
        if public_id in sequences_dict:
            seq = sequences_dict[public_id]
            results.append(SeqIO.SeqRecord(seq, id=public_id, description=""))
        else:
            print(f"Warning: Sequence for {public_id} not found in the fasta file.")

    output_path = args.output_dir + "/sequences_" + "_".join(args.store_only) if args.store_only else "all"
    output_path += ".fasta"
    print(f"Writing sequences to {output_path}...")
    with open(output_path, "w") as f:
        SeqIO.write(results, f, "fasta")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate pairwise distances.")
    parser.add_argument("--fasta", type=str, required=True, help="Path to the fasta file.")
    parser.add_argument("--labels", type=str, required=True, help="Path to the labels file.")
    parser.add_argument("--store_only", nargs="+", default=None, help="Store only these serotypes.")
    parser.add_argument("--output_dir", type=str, default="extracted_sequences", help="Path to the output directory.")
    
    args = parser.parse_args()


    main(args)