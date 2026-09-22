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

Two evaluation protocols:

  - **holdout** (pass ``--test_labels`` + ``--test_embedding_dir``): fit on the
    whole training split, predict the held-out test split. This is the protocol
    to quote against ALLCAPS, whose headline test numbers come from the same
    split via ``process_trihead_query`` + ``eval_test_performance``. Test rows
    are filtered exactly as ``eval_test_performance`` filters them (drop NON-CBL,
    serogroup-only and compound labels; keep rare serotypes), so the two
    denominators match.

  - **cross-validation** (the default, when no test set is given): out-of-fold
    predictions over the training split with ``StratifiedGroupKFold`` grouped by
    ``Public_ID``. Useful for a quick read on the training split, but it is *not*
    the same protocol as the ALLCAPS test numbers — don't put the two in one
    table without saying so.

Grouping matters in both modes: a *cps* locus is frequently split across two
contigs of one assembly, so an ungrouped fold puts near-duplicate siblings on
both sides and inflates the baseline.

The training class set (capsulated, resolved serotype, ``count >=
MIN_SEROTYPE_COUNT``) mirrors the one built in
``trihead/train_trihead_transformer.py``.

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
from sklearn.model_selection import StratifiedGroupKFold
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
    DEFAULT_NONCBL_LABEL,
    MIN_SEROTYPE_COUNT,
    SEROGROUP_LABELS,
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


def evaluate_mean(X, y, groups, class_names, params):
    """Contig-level LR on mean-pooled embeddings, out-of-fold predictions.

    Folds are grouped by ``groups`` (``Public_ID``), so every contig of one
    assembly stays on the same side. Ungrouped, the sibling contigs of a locus
    split across two contigs would sit on both sides and inflate the score.

    Returns the out-of-fold argmax prediction and the predicted-class
    probability (``serotype_confidence`` analogue) for every contig.
    """
    X = np.stack(X)
    skf = StratifiedGroupKFold(
        n_splits=params["k_folds"], shuffle=True, random_state=params["random_state"]
    )
    y_pred = np.empty_like(y)
    y_conf = np.zeros(len(y), dtype=float)
    for fold, (tr, te) in enumerate(skf.split(X, y, groups=groups), 1):
        clf = LogisticRegression(
            C=params["C"],
            max_iter=params["max_iter"],
            class_weight="balanced",
        )
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])
        y_pred[te] = clf.classes_[proba.argmax(axis=1)]
        y_conf[te] = proba.max(axis=1)
        logger.info("[mean] fold %d/%d done", fold, params["k_folds"])
    return y_pred, y_conf


def evaluate_chunk(X, y, groups, class_names, params):
    """Chunk-level LR ("LR on chunks"): every chunk is an instance, per-chunk
    class probabilities are averaged back to a contig-level prediction. Folds
    are grouped by ``groups`` (``Public_ID``) so neither the chunks of a contig
    nor the sibling contigs of an assembly straddle the split."""
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


def _fit_lr(X, y, params):
    clf = LogisticRegression(
        C=params["C"],
        max_iter=params["max_iter"],
        class_weight="balanced",
        # Two removed kwargs, both no-ops here: `multi_class="multinomial"` (gone
        # in scikit-learn 1.7 — multinomial is what lbfgs does anyway) and
        # `n_jobs` (never applied to the lbfgs solver; warns from 1.8).
    )
    clf.fit(X, y)
    return clf


def _expand_to_chunks(X, y, indices=None):
    """Per-contig (L, D) arrays -> per-chunk instances carrying the contig label."""
    indices = range(len(X)) if indices is None else indices
    Xc = np.concatenate([X[i] for i in indices], axis=0)
    yc = np.concatenate([np.full(len(X[i]), y[i]) for i in indices])
    return Xc, yc


def _probs_to_full(proba_row, clf_classes, n_classes):
    """Scatter an LR probability vector back into the full class space.

    The classifier only knows the classes it was fit on. In holdout mode the
    test set can carry a serotype the training split never had; such a row can
    never be predicted correctly, and counts as an error — which is exactly how
    ``eval_test_performance`` scores ALLCAPS, so the comparison stays fair.
    """
    full = np.zeros(n_classes)
    full[clf_classes] = proba_row
    return full


def evaluate_holdout(X_train, y_train, X_test, class_names, params):
    """Fit on the whole training split, predict the held-out test split.

    This is the protocol that matches the ALLCAPS test numbers: one model, fit
    once on train, scored on a split it has never seen.
    """
    n_classes = len(class_names)
    if params["pooling"] == "mean":
        clf = _fit_lr(np.stack(X_train), y_train, params)
        proba = clf.predict_proba(np.stack(X_test))
        full = np.zeros((len(X_test), n_classes))
        full[:, clf.classes_] = proba
    else:
        Xc, yc = _expand_to_chunks(X_train, y_train)
        logger.info("[chunk] fitting on %d chunk instances", len(Xc))
        clf = _fit_lr(Xc, yc, params)
        full = np.stack([
            _probs_to_full(clf.predict_proba(x).mean(axis=0), clf.classes_, n_classes)
            for x in X_test
        ])
    logger.info("Fitted on %d training contigs across %d classes; predicted %d test contigs",
                len(y_train), len(clf.classes_), len(X_test))
    return full.argmax(axis=1), full.max(axis=1)


def write_query_results(sample_ids, y, y_pred, y_conf, class_names, out_dir):
    """Save per-contig predictions, mirroring the trihead
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
        f.write(f"Protocol:            {params['protocol']}\n")
        if params["protocol"] == "holdout":
            f.write("Evaluated on:        held-out test split (fit once on train)\n")
        else:
            f.write(f"CV folds:            {params['k_folds']} "
                    f"(StratifiedGroupKFold, grouped by Public_ID)\n")
            f.write("Evaluated on:        training split, out-of-fold\n")
        f.write(f"Contigs evaluated:   {len(y)}\n")
        f.write(f"Serotype classes:    {len(class_names)}\n")
        f.write(f"Accuracy:            {acc:.4f}\n")
        f.write(f"Balanced accuracy:   {bal:.4f}\n")
        f.write(f"F1 (weighted):       {f1_w:.4f}\n")
        f.write(f"F1 (macro):          {f1_m:.4f}\n\n")
        kind = "holdout" if params["protocol"] == "holdout" else "out-of-fold"
        f.write(f"Classification Report ({kind}):\n")
        f.write(str(clf_report))

    logger.info("Baseline serotype results:")
    logger.info("  accuracy=%.4f  balanced=%.4f  f1_w=%.4f  f1_m=%.4f", acc, bal, f1_w, f1_m)
    logger.info("Report:      %s", report_path)
    logger.info("Confusion:   %s", cm_path)


def _read_labels(path):
    return pd.read_csv(path, index_col=0, sep="\t" if path.endswith(".tsv") else ",")


def select_rows(labels_df, label_column, missing_label, min_count, split_name):
    """Restrict a metadata table to rows a serotype classifier can be scored on.

    Keeps capsulated contigs whose label is a *resolved* serotype — dropping
    NON-CBL, serogroup-only (``Serogroup 24``) and compound (``15B/15C``)
    labels. This is the same exclusion ``eval_test_performance`` applies to
    ALLCAPS, so both models are scored over identical rows.

    ``min_count`` is applied to the **training** split only (pass ``None`` for
    test). Dropping rare classes from the training set mirrors the TriHead's
    class construction; dropping them from the test set would quietly delete the
    hardest rows and flatter both models.
    """
    n0 = len(labels_df)
    labels_df = labels_df[labels_df[label_column].notna()].copy()
    labels_df["Serotype"] = labels_df[label_column].fillna(missing_label)
    labels_df = labels_df[labels_df["Serotype"] != missing_label]
    labels_df = labels_df[labels_df["Is_capsule"].astype(bool)]
    labels_df = labels_df[labels_df["Serotype"] != DEFAULT_NONCBL_LABEL]
    labels_df = labels_df[~labels_df["Serotype"].isin(SEROGROUP_LABELS)]
    labels_df = labels_df[
        labels_df["Serotype"].map(lambda x: classify_label_type(x) == "serotype")
    ]

    if min_count is not None:
        counts = Counter(labels_df["Serotype"])
        rare = {c for c, n in counts.items() if n < min_count}
        if rare:
            logger.info("[%s] dropping %d serotypes below min_count=%d: %s",
                        split_name, len(rare), min_count, sorted(rare))
            labels_df = labels_df[~labels_df["Serotype"].isin(rare)]

    logger.info("[%s] %d/%d contigs retained across %d serotypes",
                split_name, len(labels_df), n0, labels_df["Serotype"].nunique())
    return labels_df


def load_split(embedding_dir, labels_df, pooling, split_name):
    """Metadata rows -> (sample_ids, serotypes, groups, X), dropping rows with no
    embedding file. ``groups`` is the assembly id, for sample-grouped folds."""
    sample_ids = get_sample_id(labels_df).tolist()
    serotypes = labels_df["Serotype"].to_numpy()
    groups = labels_df.index.to_numpy()

    kept_mask, X = load_pooled_embeddings(embedding_dir, sample_ids, pooling)
    if not kept_mask.all():
        logger.warning("[%s] %d/%d contigs had no embedding file in %s and were dropped",
                       split_name, (~kept_mask).sum(), len(kept_mask), embedding_dir)
    sample_ids = [sid for sid, keep in zip(sample_ids, kept_mask) if keep]
    return sample_ids, serotypes[kept_mask], groups[kept_mask], X


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

    holdout = args.test_labels is not None
    if holdout and args.test_embedding_dir is None:
        raise ValueError("--test_labels requires --test_embedding_dir")
    params["protocol"] = "holdout" if holdout else "cross-validation"

    logger.info("Loading training labels from %s", args.labels)
    train_df = select_rows(
        _read_labels(args.labels), label_column, missing_label, min_count, "train"
    )
    logger.info("Loading + pooling training embeddings (mode=%s)", params["pooling"])
    train_ids, train_sero, train_groups, X_train = load_split(
        args.embedding_dir, train_df, params["pooling"], "train"
    )

    if holdout:
        logger.info("Loading test labels from %s", args.test_labels)
        # No min_count on test: see select_rows.
        test_df = select_rows(
            _read_labels(args.test_labels), label_column, missing_label, None, "test"
        )
        logger.info("Loading + pooling test embeddings (mode=%s)", params["pooling"])
        test_ids, test_sero, _, X_test = load_split(
            args.test_embedding_dir, test_df, params["pooling"], "test"
        )

        overlap = set(train_groups) & set(test_df.index)
        if overlap:
            raise ValueError(
                f"{len(overlap)} Public_IDs appear in BOTH splits (e.g. "
                f"{sorted(overlap)[:3]}). The split is leaking; refusing to "
                f"report a number from it."
            )

        # Class space spans both splits, so a test serotype the training split
        # never had is scored as an error rather than silently dropped.
        class_names = sorted(set(train_sero) | set(test_sero))
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        y_train = np.array([class_to_idx[s] for s in train_sero])
        y = np.array([class_to_idx[s] for s in test_sero])

        unseen = sorted(set(test_sero) - set(train_sero))
        if unseen:
            n_unseen = int(np.isin(test_sero, unseen).sum())
            logger.warning(
                "%d test contigs carry %d serotype(s) absent from training (%s) — "
                "unpredictable by construction, counted as errors",
                n_unseen, len(unseen), unseen,
            )

        logger.info("Holdout: fit on %d training contigs, score %d test contigs "
                    "across %d classes", len(y_train), len(y), len(class_names))
        y_pred, y_conf = evaluate_holdout(X_train, y_train, X_test, class_names, params)
        eval_ids = test_ids
    else:
        class_names = sorted(set(train_sero))
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        y = np.array([class_to_idx[s] for s in train_sero])
        logger.info("Cross-validation: %d contigs across %d serotype classes, "
                    "%d folds grouped by Public_ID",
                    len(y), len(class_names), params["k_folds"])

        if params["pooling"] == "mean":
            y_pred, y_conf = evaluate_mean(X_train, y, train_groups, class_names, params)
        else:
            y_pred, y_conf = evaluate_chunk(X_train, y, train_groups, class_names, params)
        eval_ids = train_ids

    write_reports(y, y_pred, class_names, args.output_dir, params)
    write_query_results(eval_ids, y, y_pred, y_conf, class_names, args.output_dir)


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
        "--test_labels",
        default=None,
        help="Held-out test metadata (e.g. test_metadata.csv). Switches to holdout "
        "evaluation: fit on the whole training split, score the test split. This is "
        "the protocol comparable to the ALLCAPS test numbers. Requires "
        "--test_embedding_dir. Omit for grouped cross-validation on the training split.",
    )
    parser.add_argument(
        "--test_embedding_dir",
        default=None,
        help="Directory of ProkBERT chunk embeddings for the TEST contigs. These are "
        "not produced by the training pipeline (which only embeds train.fasta) — "
        "generate them first with: python -m scripts.embed_transformer "
        "--fasta test.fasta --out_dir <dir>",
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
