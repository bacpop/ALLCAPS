#!/usr/bin/env python
"""Baseline serotype classifier: ProkBERT chunk embeddings + Logistic Regression.

This is the *pre-transformer* model, kept as a reference point to justify the
learned encoder. It reads the SAME inputs as the TriHead trainer — the flat
directory of per-sample ``(L, D)`` ProkBERT chunk embeddings and the metadata
CSV — but replaces the learned ``TransformerEncoder`` aggregation with a plain
pooling step (mean over chunks) followed by a multinomial Logistic Regression.
The only thing that differs from the transformer pipeline is the aggregation
head, so any accuracy gap is attributable to the encoder.

Two pooling modes (``pooling`` in ``--model_params``):
  - ``mean``  (default): mean-pool the L chunks to one (D,) vector per sample,
    then LR at the sample level. Fair, direct analogue of the encoder's
    masked-mean pooling but without attention.
  - ``chunk``: treat every chunk as its own training instance carrying the
    sample's label ("LR on chunks"), then average per-chunk class
    probabilities back to a sample-level prediction. Grouped folds
    (StratifiedGroupKFold) keep all chunks of a sample on the same side.

Evaluation is out-of-fold (StratifiedKFold, k=5 by default) over capsulated,
resolved-serotype samples with ``count >= MIN_SEROTYPE_COUNT`` — mirroring the
class set built in ``trihead/train_trihead_transformer.py`` so the numbers are
directly comparable to ``eval_serotype_classifier.py``.

Reports are written to ``--output_dir`` with a ``baseline_lr_`` prefix so they
never clobber the transformer's outputs in a shared results directory.
"""

import os
import json
import argparse
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)

from .consts import (
    RND_STATE,
    DEFAULT_KFOLDS,
    DEFAULT_LABEL_COLUMN,
    DEFAULT_MISSING_LABEL,
    MIN_SEROTYPE_COUNT,
)
from .logging_config import get_logger
from .utils import get_sample_id, classify_label_type

logger = get_logger(__name__)

DEFAULT_C = 1.0
DEFAULT_MAX_ITER = 2000
DEFAULT_POOLING = "mean"


def load_pooled_embeddings(embedding_dir, sample_ids, pooling):
    """Load per-sample chunk embeddings and (optionally) mean-pool them.

    Returns
    -------
    kept_mask : np.ndarray[bool]  -- which sample_ids had an embedding file.
    X         : list              -- one entry per kept sample. For pooling
                                     ``mean`` each entry is a (D,) vector; for
                                     ``chunk`` each entry is the raw (L, D)
                                     array (expanded to instances later).
    """
    kept_mask = np.zeros(len(sample_ids), dtype=bool)
    X = []
    for i, sid in enumerate(sample_ids):
        path = os.path.join(embedding_dir, f"{sid}.npy")
        if not os.path.exists(path):
            continue
        chunks = np.load(path)  # (L, D)
        if chunks.ndim == 1:  # single-chunk fallback saved as (D,)
            chunks = chunks[None, :]
        kept_mask[i] = True
        X.append(chunks.mean(axis=0) if pooling == "mean" else chunks)
    return kept_mask, X


def evaluate_mean(X, y, class_names, params):
    """Sample-level LR on mean-pooled embeddings, out-of-fold predictions.

    Returns the out-of-fold argmax prediction and the predicted-class
    probability (``serotype_confidence`` analogue) for every sample.
    """
    X = np.stack(X)
    skf = StratifiedKFold(
        n_splits=params["k_folds"], shuffle=True, random_state=params["random_state"]
    )
    y_pred = np.empty_like(y)
    y_conf = np.zeros(len(y), dtype=float)
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        clf = LogisticRegression(
            C=params["C"],
            max_iter=params["max_iter"],
            class_weight="balanced",
            multi_class="multinomial",
            n_jobs=-1,
        )
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])
        y_pred[te] = clf.classes_[proba.argmax(axis=1)]
        y_conf[te] = proba.max(axis=1)
        logger.info("[mean] fold %d/%d done", fold, params["k_folds"])
    return y_pred, y_conf


def evaluate_chunk(X, y, groups, class_names, params):
    """Chunk-level LR ("LR on chunks"): every chunk is an instance, per-chunk
    class probabilities are averaged back to a sample-level prediction. Folds
    are grouped by sample so no sample straddles the train/test split."""
    n_classes = len(class_names)
    sgkf = StratifiedGroupKFold(
        n_splits=params["k_folds"], shuffle=True, random_state=params["random_state"]
    )
    y_pred = np.empty_like(y)
    y_conf = np.zeros(len(y), dtype=float)
    for fold, (tr, te) in enumerate(sgkf.split(X, y, groups=groups), 1):
        # Expand training samples to per-chunk instances.
        Xtr = np.concatenate([X[i] for i in tr], axis=0)
        ytr = np.concatenate([np.full(len(X[i]), y[i]) for i in tr])
        clf = LogisticRegression(
            C=params["C"],
            max_iter=params["max_iter"],
            class_weight="balanced",
            multi_class="multinomial",
            n_jobs=-1,
        )
        clf.fit(Xtr, ytr)
        # Predict per test sample by averaging its chunks' class probabilities.
        for i in te:
            probs = clf.predict_proba(X[i]).mean(axis=0)  # (n_present_classes,)
            full = np.zeros(n_classes)
            full[clf.classes_] = probs
            y_pred[i] = full.argmax()
            y_conf[i] = full.max()
        logger.info("[chunk] fold %d/%d done", fold, params["k_folds"])
    return y_pred, y_conf


def write_query_results(sample_ids, y, y_pred, y_conf, class_names, out_dir):
    """Save per-sample out-of-fold predictions, mirroring the trihead
    ``query_results.csv`` layout (index = record id, ``pred_argmax`` +
    ``serotype_confidence``). The baseline has no CBL / novelty / genogroup
    heads, so only the applicable columns are written, plus the ground-truth
    ``serotype`` for direct comparison."""
    results_df = pd.DataFrame(
        {
            "serotype": [class_names[i] for i in y],
            "serotype_confidence": np.round(y_conf, 3),
            "pred_argmax": [class_names[i] for i in y_pred],
        },
        index=sample_ids,
    )
    out_path = os.path.join(out_dir, "baseline_lr_query_results.csv")
    results_df.to_csv(out_path)
    logger.info("Query results: %s", out_path)


def write_reports(y, y_pred, class_names, out_dir, params):
    target_idx = sorted(set(y.tolist()) | set(y_pred.tolist()))
    target_names = [class_names[i] for i in target_idx]

    acc = accuracy_score(y, y_pred)
    f1_w = f1_score(y, y_pred, average="weighted")
    f1_m = f1_score(y, y_pred, average="macro")
    bal = balanced_accuracy_score(y, y_pred)
    clf_report = classification_report(
        y, y_pred, labels=target_idx, target_names=target_names, zero_division=0
    )

    cm = confusion_matrix(y, y_pred, labels=target_idx)
    cm_path = os.path.join(out_dir, "baseline_lr_confusion_matrix_df.csv")
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(cm_path)

    report_path = os.path.join(out_dir, "baseline_lr_classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Baseline (ProkBERT chunks + Logistic Regression) Serotype Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Pooling mode:        {params['pooling']}\n")
        f.write(f"CV folds:            {params['k_folds']}\n")
        f.write(f"Samples evaluated:   {len(y)}\n")
        f.write(f"Serotype classes:    {len(class_names)}\n")
        f.write(f"Accuracy:            {acc:.4f}\n")
        f.write(f"Balanced accuracy:   {bal:.4f}\n")
        f.write(f"F1 (weighted):       {f1_w:.4f}\n")
        f.write(f"F1 (macro):          {f1_m:.4f}\n\n")
        f.write("Classification Report (out-of-fold):\n")
        f.write(str(clf_report))

    logger.info("Baseline serotype results:")
    logger.info("  accuracy=%.4f  balanced=%.4f  f1_w=%.4f  f1_m=%.4f", acc, bal, f1_w, f1_m)
    logger.info("Report:      %s", report_path)
    logger.info("Confusion:   %s", cm_path)


def main(args):
    params = args.model_params
    params["k_folds"] = params.get("k_folds", DEFAULT_KFOLDS)
    params["random_state"] = params.get("random_state", RND_STATE)
    params["C"] = params.get("C", DEFAULT_C)
    params["max_iter"] = params.get("max_iter", DEFAULT_MAX_ITER)
    params["pooling"] = params.get("pooling", DEFAULT_POOLING)
    label_column = params.get("label_column", DEFAULT_LABEL_COLUMN)
    missing_label = params.get("missing_label", DEFAULT_MISSING_LABEL)
    min_count = params.get("min_serotype_count", MIN_SEROTYPE_COUNT)

    if params["pooling"] not in ("mean", "chunk"):
        raise ValueError(f"Unknown pooling mode: {params['pooling']}")

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading labels from %s", args.labels)
    labels_df = pd.read_csv(
        args.labels, index_col=0, sep="\t" if args.labels.endswith(".tsv") else ","
    )
    labels_df["Serotype"] = labels_df[label_column].fillna(missing_label)

    # ── Class set: capsulated + resolved serotype + count >= min_count ──
    #   (mirrors trihead/train_trihead_transformer.py:504-539)
    labels_df = labels_df[labels_df["Serotype"] != missing_label]
    labels_df = labels_df[labels_df["Is_capsule"].astype(bool)]
    labels_df = labels_df[
        labels_df["Serotype"].map(lambda s: classify_label_type(s) == "serotype")
    ]
    counts = Counter(labels_df["Serotype"])
    rare = {s for s, c in counts.items() if c < min_count}
    if rare:
        logger.info("Dropping %d serotypes below min_count=%d: %s",
                    len(rare), min_count, sorted(rare))
        labels_df = labels_df[~labels_df["Serotype"].isin(rare)]

    sample_ids = get_sample_id(labels_df).tolist()
    serotypes = labels_df["Serotype"].to_numpy()

    logger.info("Loading + pooling embeddings (mode=%s) for %d samples",
                params["pooling"], len(sample_ids))
    kept_mask, X = load_pooled_embeddings(args.embedding_dir, sample_ids, params["pooling"])
    if not kept_mask.all():
        logger.warning("%d/%d samples had no embedding file and were dropped",
                       (~kept_mask).sum(), len(kept_mask))
    serotypes = serotypes[kept_mask]
    sample_ids = [sid for sid, keep in zip(sample_ids, kept_mask) if keep]

    class_names = sorted(set(serotypes))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    y = np.array([class_to_idx[s] for s in serotypes])
    logger.info("Evaluating %d samples across %d serotype classes",
                len(y), len(class_names))

    if params["pooling"] == "mean":
        y_pred, y_conf = evaluate_mean(X, y, class_names, params)
    else:
        groups = np.arange(len(y))  # one group per sample
        y_pred, y_conf = evaluate_chunk(X, y, groups, class_names, params)

    write_reports(y, y_pred, class_names, args.output_dir, params)
    write_query_results(sample_ids, y, y_pred, y_conf, class_names, args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline serotype classifier (ProkBERT chunks + Logistic Regression)."
    )
    parser.add_argument(
        "--embedding_dir",
        required=True,
        help="Directory of per-sample (L, D) ProkBERT chunk embeddings (.npy). "
        "Same directory passed to the TriHead trainer as --embedding_dir.",
    )
    parser.add_argument(
        "--labels", required=True, help="Metadata CSV/TSV (e.g. final_metadata.csv)."
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory for the baseline reports."
    )
    parser.add_argument(
        "--model_params",
        type=str,
        default="{}",
        help='JSON: pooling ("mean"|"chunk"), k_folds, C, max_iter, '
        "min_serotype_count, random_state, label_column, missing_label.",
    )
    args = parser.parse_args()
    try:
        args.model_params = json.loads(args.model_params)
        if not isinstance(args.model_params, dict):
            logger.warning("model_params must be a JSON object; using defaults.")
            args.model_params = {}
    except json.JSONDecodeError:
        logger.error("Could not parse model_params JSON; using defaults.")
        args.model_params = {}
    return args


if __name__ == "__main__":
    main(parse_args())
