#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score

MISSING_LABEL = "Non-typeable"  # TODO: Make this a parameter
DEFAULT_TEST_SIZE = 0.2
DEFAULT_KNN_K = 5
MIN_COUNT = 2  # Minimum count for a label to be considered valid

def main(args):
    print(f"Loading contrastive embeddings and labels...")
    X = np.load(args.embeddings)  # shape (N, new_dim)
    labels = pd.read_csv(args.labels, sep="\t", index_col=0)
    assert X.shape[0] == len(labels), "Number of embeddings and labels do not match."

    labels['Serotype'] = labels['Serotype'].fillna(MISSING_LABEL)
    known_indices = labels['Serotype'] != MISSING_LABEL
    
    # Drop samples with labels that only occur once
    underrep_labels = labels['Serotype'].value_counts()[labels['Serotype'].value_counts() < MIN_COUNT].index
    if underrep_labels.any():
        print("Dropping serotypes with only one sample:", *underrep_labels.to_list())
    known_indices &= ~labels['Serotype'].isin(underrep_labels)
    
    X_known, labels_known = X[known_indices], labels[known_indices]
    print(f"Total samples: {X.shape[0]}, known-label samples: {X_known.shape[0]}")

    print(f"Training k-NN with k={args.knn_k} on final embeddings...")
    X_train, X_test, y_train, y_test = train_test_split(X_known, labels_known, test_size=args.test_size, random_state=42, stratify=labels_known)
    knn = KNeighborsClassifier(n_neighbors=args.knn_k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    report_str = classification_report(y_test, y_pred)
    f1_str = f"F1-Weighted: {f1_score(y_test, y_pred, average='weighted')}"
    print("Classification Report:")
    print(report_str)
    print(f1_str)

    with open(args.output, "w") as f:
        f.write("Classification Report (test set)\n")
        f.write(report_str + "\n")
        f.write(f1_str)

    print(f"Saved k-NN classification report to {args.output}")

def parse_args():
    parser = argparse.ArgumentParser(description="KNN serotype inference on precomputed contrastive embeddings.")
    parser.add_argument("--embeddings", required=True, help="Path to the final contrastive embeddings (.npy) of shape (N, new_dim).")
    parser.add_argument("--labels", required=True, help="Path to labels")
    parser.add_argument("--knn_k", type=int, default=DEFAULT_KNN_K, help="Number of neighbors for k-NN.")
    parser.add_argument("--test_size", type=float, default=DEFAULT_TEST_SIZE, help="Fraction of data for test split.")
    parser.add_argument("--output", required=True, help="Where to save the text classification report.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
