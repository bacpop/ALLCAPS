import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

from consts import DEFAULT_SEP, DEFAULT_LABEL_COLUMN, DEFAULT_MISSING_LABEL

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate pairwise distances.")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to the embeddings file.")
    parser.add_argument("--labels", type=str, required=True, help="Path to the labels file.")
    parser.add_argument("--output", type=str, default="distances.npz", help="Path to the output file.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it exists.")
    return parser.parse_args()


def main(args):
    ### TODO move to params
    sep = DEFAULT_SEP
    label_column = DEFAULT_LABEL_COLUMN
    missing_label = DEFAULT_MISSING_LABEL
    ###
    print("Loading embeddings...")
    X = np.load(args.embeddings)
    labels = pd.read_csv(args.labels, index_col=0, sep="\t")

    labels[label_column] = labels[label_column].fillna(missing_label)
    indices = labels[label_column] != missing_label
    labels = labels[indices][label_column]

    is_emb_npz = isinstance(X, np.lib.npyio.NpzFile)
    if is_emb_npz:
        cbl_prefix = lambda k: f"cbl{sep}{k}"
        X = np.array([X[cbl_prefix(key)] for key in labels.index])
    
    if os.path.exists(args.output):
        print(f"Output file {args.output} already exists.", end=" ")
        if args.overwrite:
            print("Overwriting...")
        else:
            print("Exiting without overwriting.")
            return
        
    print("Calculating pairwise distances...")
    dist_matrix = cosine_distances(X)
    pairwise_distances = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    
    np.savez(args.output, distances=pairwise_distances, size=dist_matrix.shape[0])
    print(f"Pairwise distances saved to {args.output}.")


if __name__ == "__main__":
    main(parse_args())
