#!/usr/bin/env python

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler

MISSING_LABEL = "Non-typeable"  # TODO: Make this a parameter
DEFAULT_TEST_SIZE = 0.2
DEFAULT_KNN_K = 5
MIN_COUNT = 2  # Minimum count for a label to be considered valid


def main(args):
    print(f"Loading contrastive embeddings and labels...")
    X = np.load(args.embeddings)  # shape (N, new_dim)
    is_X_npz = isinstance(X, np.lib.npyio.NpzFile)
    labels = pd.read_csv(args.labels, sep="\t")
    
    if not is_X_npz:
        assert len(X) == len(labels), "Number of embeddings and labels do not match."
        # TODO add sanity test for npz


    labels['Serotype'] = labels['Serotype'].fillna(MISSING_LABEL)

    # In case "someone" has messed up while cleaning the data
    is_duplicate = labels.duplicated()
    if is_duplicate.any():
        print("Dropping duplicate label rows...")
        labels = labels[~is_duplicate]

    known_indices = labels['Serotype'] != MISSING_LABEL

    # Drop samples with labels that only occur once
    underrep_labels = labels['Serotype'].value_counts()[labels['Serotype'].value_counts() < MIN_COUNT].index
    if underrep_labels.any():
        print(f"Dropping serotypes with less than {MIN_COUNT} samples:", *underrep_labels.to_list())
        known_indices &= ~labels['Serotype'].isin(underrep_labels)

    labels_known = labels[known_indices]["Serotype"]
    if is_X_npz:
        X_known = np.array([X[key] for key in labels[known_indices]["Public_name"]])
    else:
        X = X[~is_duplicate]
        X_known = X[known_indices]
    print(f"Known-label samples: {X_known.shape[0]}")

    print(f"Training k-NN with k={args.knn_k} on standardized embeddings...")
    
    X_known = StandardScaler().fit_transform(X_known)
    X_train, X_test, y_train, y_test = train_test_split(X_known, labels_known, test_size=args.test_size,
                                                        random_state=42, stratify=labels_known)
    knn = KNeighborsClassifier(n_neighbors=args.knn_k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Storing the classification results
    report_str = classification_report(y_test, y_pred)
    f1_str = f"F1-Weighted: {f1_score(y_test, y_pred, average='weighted')}"
    print("Classification Report:")
    print(report_str)
    print(f1_str)

    with open(Path(args.output_dir) / "knn_report.txt", "w") as f:
        f.write("Classification Report (test set)\n")
        f.write(report_str + "\n")
        f.write(f1_str)

    conf_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    conf_matrix = conf_matrix.reindex(index=conf_matrix.index, columns=conf_matrix.index, fill_value=0)
    conf_matrix.to_csv(Path(args.output_dir) / "confusion_matrix_df.csv")

    print(f"Saved k-NN classification report to {args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="KNN serotype inference on precomputed contrastive embeddings.")
    parser.add_argument("--embeddings", required=True,
                        help="Path to the final contrastive embeddings (.npy/.npz) of shape (N, new_dim).")
    parser.add_argument("--labels", required=True, help="Path to labels")
    parser.add_argument("--knn_k", type=int, default=DEFAULT_KNN_K, help="Number of neighbors for k-NN.")
    parser.add_argument("--test_size", type=float, default=DEFAULT_TEST_SIZE, help="Fraction of data for test split.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the text classification report in.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
