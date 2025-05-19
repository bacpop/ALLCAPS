import os
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_distances


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate pairwise distances.")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to the embeddings file.")
    parser.add_argument("--labels", type=str, required=True, help="Path to the labels file.")
    parser.add_argument("--output", type=str, default="distances.npz", help="Path to the output file.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it exists.")
    return parser.parse_args()


def main(args):
    print("Loading embeddings...")
    embeddings = np.load(args.embeddings)
    
    if os.path.exists(args.output):
        print(f"Output file {args.output} already exists.", end=" ")
        if args.overwrite:
            print("Overwriting...")
        else:
            print("Exiting without overwriting.")
            return
        
    print("Calculating pairwise distances...")
    dist_matrix = cosine_distances(embeddings)
    pairwise_distances = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    
    np.savez(args.output, distances=pairwise_distances, size=dist_matrix.shape[0])
    print(f"Pairwise distances saved to {args.output}.")


if __name__ == "__main__":
    main(parse_args())
