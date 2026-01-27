#!/usr/bin/env python
"""Evaluate linear/SVM separability on learned embeddings."""

import argparse
import json
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from .consts import DEFAULT_MISSING_LABEL, DEFAULT_SEP, DEFAULT_LABEL_COLUMN, RND_STATE
from .utils import get_sample_id

SVM_TEST_SIZE = 0.2
SVM_C = 1.0
SVM_KERNEL = "linear"
SVM_GAMMA = "scale"
SVM_DEGREE = 3

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SVM sanity check on embeddings")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings .npz (from infer_transformer)")
    parser.add_argument("--labels", required=True, help="Path to labels TSV (e.g., final_metadata.tsv)")
    parser.add_argument("--output", required=True, help="Path to write TXT report")
    parser.add_argument("--task", choices=["capsule", "serotype", "both"], default="both",
                        help="Which task to evaluate: capsule vs non-capsule, serotype classification, or both")
    parser.add_argument("--model_params", type=str, default="{}",
                        help="JSON string; supports svm_C, svm_kernel, svm_gamma, svm_degree, test_size, random_state, sep, label_column, missing_label")
    args = parser.parse_args()

    try:
        args.model_params = json.loads(args.model_params)
        if not isinstance(args.model_params, dict):
            print("model_params must be a JSON object; falling back to defaults.")
            args.model_params = {}
    except json.JSONDecodeError:
        print("Could not parse model_params JSON; using defaults.")
        args.model_params = {}

    return args


def load_embeddings_and_labels(args: argparse.Namespace) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, np.ndarray]]:
    sep = args.model_params.get("sep", DEFAULT_SEP)
    label_column = args.model_params.get("label_column", DEFAULT_LABEL_COLUMN)
    missing_label = args.model_params.get("missing_label", DEFAULT_MISSING_LABEL)

    labels_df = pd.read_csv(args.labels, sep="\t", index_col=0)
    labels_df["Serotype"] = labels_df[label_column].fillna(missing_label)

    npz = np.load(args.embeddings, allow_pickle=True)

    # Build keys consistent with infer_transformer output
    keys = labels_df["Is_capsule"].map(lambda x: "cbl" if x else "non-cbl") + sep + get_sample_id(labels_df)
    available = []
    embeddings = []
    for k in keys:
        if k in npz:
            embeddings.append(npz[k])
            available.append(True)
        else:
            available.append(False)
    mask = pd.Series(available, index=labels_df.index)
    if not mask.all():
        missing = (~mask).sum()
        print(f"Warning: {missing} entries missing embeddings; they will be dropped.")
    labels_df = labels_df[mask]
    X = np.stack(embeddings)
    return X, labels_df, npz


def evaluate_capsule(X: np.ndarray, labels_df: pd.DataFrame, params: Dict) -> Dict[str, Any]:
    y = labels_df["Is_capsule"].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], stratify=y, random_state=params["random_state"]
    )
    clf = SVC(C=params["svm_C"], kernel=params["svm_kernel"], gamma=params["svm_gamma"], degree=params["svm_degree"])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return {
        "task": "capsule",
        "n_train": len(y_train),
        "n_test": len(y_test),
        "acc": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
        "balanced_acc": balanced_accuracy_score(y_test, y_pred),
    }


def evaluate_serotype(X: np.ndarray, labels_df: pd.DataFrame, params: Dict) -> Dict[str, Any]:
    cap_mask = labels_df["Is_capsule"].astype(bool)
    labels_capsule = labels_df[cap_mask]
    X_capsule = X[cap_mask.to_numpy()]

    # Drop missing labels if any
    serotypes = labels_capsule["Serotype"].to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(serotypes)

    # Drop classes with too few samples
    unique, counts = np.unique(y, return_counts=True)
    valid_classes = unique[counts >= 2]
    valid_mask = np.isin(y, valid_classes)
    y = y[valid_mask]
    X_capsule = X_capsule[valid_mask]
    print(f"Dropping {len(unique) - len(valid_classes)} classes with <2 samples:", le.inverse_transform(unique[counts < 2]))

    X_train, X_test, y_train, y_test = train_test_split(
        X_capsule, y, test_size=params["test_size"], stratify=y, random_state=params["random_state"]
    )
    clf = SVC(C=params["svm_C"], kernel=params["svm_kernel"], gamma=params["svm_gamma"], degree=params["svm_degree"])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return {
        "task": "serotype",
        "n_classes": len(le.classes_),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "acc": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
        "balanced_acc": balanced_accuracy_score(y_test, y_pred),
    }


def format_report(results: Dict[str, Any]) -> str:
    lines = [
        f"Task: {results.get('task')}",
    ]
    if results.get("n_classes"):
        lines.append(f"Classes: {results['n_classes']}")
    lines.extend([
        f"Train samples: {results.get('n_train')}",
        f"Test samples:  {results.get('n_test')}",
        f"Accuracy:      {results.get('acc'):.4f}",
        f"Macro F1:      {results.get('macro_f1'):.4f}",
        f"Weighted F1:   {results.get('weighted_f1'):.4f}",
        f"Balanced Acc:  {results.get('balanced_acc'):.4f}",
        "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    X, labels_df, _ = load_embeddings_and_labels(args)

    params = args.model_params
    params["test_size"] = params.get("test_size", SVM_TEST_SIZE)
    params["random_state"] = params.get("random_state", RND_STATE)
    params["svm_C"] = params.get("svm_C", SVM_C)
    params["svm_kernel"] = params.get("svm_kernel", SVM_KERNEL)
    params["svm_gamma"] = params.get("svm_gamma", SVM_GAMMA)
    params["svm_degree"] = params.get("svm_degree", SVM_DEGREE)

    tasks = [args.task] if args.task != "both" else ["capsule", "serotype"]

    reports = []
    for task in tasks:
        if task == "capsule":
            res = evaluate_capsule(X, labels_df, params)
        else:
            res = evaluate_serotype(X, labels_df, params)
        report_txt = format_report(res)
        print(report_txt)
        reports.append(report_txt)

    with open(args.output, "w") as f:
        f.write("\n".join(reports))
    print(f"\nSaved SVM sanity check report to {args.output}")


if __name__ == "__main__":
    main()
