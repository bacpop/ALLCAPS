import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, f1_score

MISSING_LABEL = "Non-typeable"  # TODO: Make this a parameter
DEFAULT_TEST_SIZE = 0.2
DEFAULT_COMPONENTS = 5
MIN_COUNT = 2  # Minimum count for a label to be considered valid
CV = 5  # Number of cross-validation folds
RND_STATE = 42

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RF on LDA-reduced transformer embeddings."
    )
    parser.add_argument('--embeddings', type=str, required=True)
    parser.add_argument('--labels', type=str, required=True)
    parser.add_argument('--n_components', type=int, default=DEFAULT_COMPONENTS)
    parser.add_argument('--test_size', type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument('--output-dir', required=True, type=str)

    return parser.parse_args()


def main(args):
    print(f"Loading embeddings and labels...")
    X = np.load(args.embeddings)  # shape (N, new_dim)
    labels = pd.read_csv(args.labels, sep="\t")
    assert X.shape[0] == len(labels), "Number of embeddings and labels do not match."

    labels['Serotype'] = labels['Serotype'].fillna(MISSING_LABEL)

    is_duplicate = labels.duplicated()
    if is_duplicate.any():
        print("Dropping duplicate label rows...")
        labels, X = labels[~is_duplicate], X[~is_duplicate]

    known_indices = labels['Serotype'] != MISSING_LABEL

    # Drop samples with labels that only occur once
    underrep_labels = labels['Serotype'].value_counts()[labels['Serotype'].value_counts() < MIN_COUNT].index
    if underrep_labels.any():
        print(f"Dropping serotypes with less than {MIN_COUNT} samples:", *underrep_labels.to_list())
        known_indices &= ~labels['Serotype'].isin(underrep_labels)

    X_known, labels_known = X[known_indices], labels[known_indices]["Serotype"]
    print(f"Total samples: {X.shape[0]}, known-label samples: {X_known.shape[0]}")

    X_train, X_test, y_train, y_test = train_test_split(X_known, labels_known, test_size=args.test_size,
                                                        random_state=RND_STATE, stratify=labels_known)

    print("Training LDA and Random Forest...")
    print("\tStandardizing data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\tApplying LDA...")
    lda = LinearDiscriminantAnalysis(n_components=args.n_components)
    X_train_lda = lda.fit_transform(X_train_scaled, y_train)
    X_test_lda = lda.transform(X_test_scaled)

    print("\tTraining Random Forest with GridSearchCV...")
    rf = RandomForestClassifier(random_state=RND_STATE)
    grid = GridSearchCV(rf, param_grid, cv=CV, n_jobs=-1, verbose=1)
    grid.fit(X_train_lda, y_train)
    best_rf = grid.best_estimator_

    print("\tEvaluating model...\n")
    y_pred = best_rf.predict(X_test_lda)
    report_str = classification_report(y_test, y_pred)
    f1_str = f"F1-Weighted: {f1_score(y_test, y_pred, average='weighted')}"

    print("Classification Report:")
    print(report_str)
    print(f1_str)

    with open(Path(args.output_dir) / "lda_rf_report.txt", "w") as f:
        f.write(f"Best parameters: {grid.best_params_}\n")
        f.write("Classification Report (test set)\n")
        f.write(report_str + "\n")
        f.write(f1_str)

    conf_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    conf_matrix = conf_matrix.reindex(index=conf_matrix.index, columns=conf_matrix.index, fill_value=0)
    conf_matrix.to_csv(Path(args.output_dir) / "lda_rf_confusion_matrix_df.csv")

    print(f"Saved reports to {args.output_dir}")


if __name__ == '__main__':
    main(parse_args())
