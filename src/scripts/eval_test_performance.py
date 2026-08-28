"""Evaluate serotype (and optional genogroup) performance from query_results.csv.

This script joins query results (from process_query.py) with metadata
(e.g., test_metadata.csv from train_test_data_split.py) and reports
classification metrics for serotype, and genogroup if present.
"""

import argparse
import os
from typing import Optional, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

from .consts import (
    DEFAULT_LABEL_COLUMN,
    DEFAULT_MISSING_LABEL,
    DEFAULT_NONCBL_LABEL,
    DEFAULT_ID_COLUMN,
	DEFAULT_CONTIG_COLUMN,
    SEROGROUP_LABELS,
    CONTIG_SEP,
)
from .logging_config import get_logger
from .utils import map_serotype_to_group, classify_label_type

logger = get_logger(__name__)

DEFAULT_PRED_COLUMN = "pred_argmax"
DEFAULT_GENOGROUP_COLUMN = "pred_genogroup"


def _read_table(path: str) -> pd.DataFrame:
    sep = "\t" if path.endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep)


def _build_metadata_id(
    meta: pd.DataFrame,
    id_column: str,
    contig_column: str,
    id_sep: str,
    metadata_id_column: Optional[str],
) -> pd.Series:
    if metadata_id_column and metadata_id_column in meta.columns:
        return meta[metadata_id_column].astype(str)
    if id_column in meta.columns and contig_column in meta.columns:
        return meta[id_column].astype(str) + id_sep + meta[contig_column].astype(str)
    if id_column in meta.columns:
        return meta[id_column].astype(str)
    raise ValueError(
        f"Could not build metadata ids. Missing columns: {id_column}, {contig_column}, or {metadata_id_column}."
    )


def _write_report(
    report_path: str,
    title: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray,
    clf_report: str,
    accuracy: float,
    f1_weighted: float,
    f1_macro: float,
):
    with open(report_path, "w") as f:
        f.write(f"{title}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Samples evaluated: {len(y_true)}\n")
        f.write(f"Unique classes: {len(labels)}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 (weighted): {f1_weighted:.4f}\n")
        f.write(f"F1 (macro): {f1_macro:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(clf_report)


def _evaluate(
    df: pd.DataFrame,
    true_col: str,
    pred_col: str,
    output_dir: str,
    report_stem: str,
    missing_label: str,
    exclude_labels: Optional[set] = None,
):
    valid_mask = df[true_col].notna() & df[pred_col].notna()
    df_eval = df.loc[valid_mask].copy()
    df_eval = df_eval[df_eval[true_col] != missing_label]
    if exclude_labels:
        n_before = len(df_eval)
        df_eval = df_eval[~df_eval[true_col].isin(exclude_labels)]
        n_excluded = n_before - len(df_eval)
        if n_excluded > 0:
            logger.info(
                "[%s] Excluded %d samples with labels: %s",
                report_stem,
                n_excluded,
                sorted(exclude_labels & set(df.loc[valid_mask, true_col].unique())),
            )
    if df_eval.empty:
        logger.warning("No valid samples for %s evaluation.", report_stem)
        return

    y_true = df_eval[true_col].astype(str).to_numpy()
    y_pred = df_eval[pred_col].astype(str).to_numpy()
    labels = np.array(sorted(list(set(y_true) | set(y_pred))))

    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    clf_report = classification_report(
        y_true, y_pred, labels=labels, target_names=labels
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(os.path.join(output_dir, f"{report_stem}_confusion_matrix.csv"))

    report_path = os.path.join(output_dir, f"{report_stem}_report.txt")
    _write_report(
        report_path=report_path,
        title=f"{report_stem} classification report",
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        clf_report=clf_report,
        accuracy=accuracy,
        f1_weighted=f1_weighted,
        f1_macro=f1_macro,
    )

    logger.info("%s accuracy: %.4f", report_stem, accuracy)
    logger.info("%s F1 (weighted): %.4f", report_stem, f1_weighted)
    logger.info("%s F1 (macro): %.4f", report_stem, f1_macro)
    logger.info("Saved %s report to: %s", report_stem, report_path)


def _build_genogroup_map(
    meta: pd.DataFrame, label_column: str, genogroup_column: str
) -> Dict[str, str]:
    if label_column not in meta.columns or genogroup_column not in meta.columns:
        return {}

    mapping = {}
    for serotype, group in meta[[label_column, genogroup_column]].dropna().values:
        serotype = str(serotype)
        group = str(group)
        if serotype in mapping and mapping[serotype] != group:
            # Resolve conflicts by choosing the most frequent mapping
            pass
        mapping.setdefault(serotype, group)

    if len(mapping) == 0:
        return {}

    # Resolve conflicts by mode per serotype
    counts = meta[[label_column, genogroup_column]].dropna()
    for serotype in counts[label_column].unique():
        subset = counts[counts[label_column] == serotype][genogroup_column]
        if not subset.empty:
            mapping[str(serotype)] = str(subset.mode().iloc[0])
    return mapping


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    query_df = pd.read_csv(args.query_results)
    if "record_id" in query_df.columns:
        query_df = query_df.set_index("record_id")
    elif (
        query_df.columns[0] != DEFAULT_PRED_COLUMN
        and query_df.columns[0] != "pred_argmax"
    ):
        query_df = query_df.set_index(query_df.columns[0])

    if args.pred_column not in query_df.columns:
        raise ValueError(
            f"Prediction column '{args.pred_column}' not found in {args.query_results}"
        )

    meta = _read_table(args.metadata)
    meta_id = _build_metadata_id(
        meta,
        id_column=args.id_column,
        contig_column=args.contig_column,
        id_sep=args.id_sep,
        metadata_id_column=args.metadata_id_column,
    )
    meta = meta.copy()
    meta["record_id"] = meta_id

    n_meta, n_query = len(meta), len(query_df)
    merged = meta.merge(query_df, left_on="record_id", right_index=True, how="inner")
    if merged.empty:
        raise ValueError("No matching records between metadata and query_results.csv")
    if len(merged) < min(n_meta, n_query):
        logger.warning(
            "Inner merge dropped samples. Meta=%d, Query=%d, Merged=%d",
            n_meta,
            n_query,
            len(merged),
        )

    label_column = args.label_column
    if label_column not in merged.columns:
        raise ValueError(f"True label column '{label_column}' not found in metadata")

    merged.rename(columns={label_column: "true_serotype"}, inplace=True)
    merged.rename(columns={args.pred_column: "pred_serotype"}, inplace=True)

    # ── Diagnostic: label-type breakdown ─────────────────────
    true_labels = merged["true_serotype"].astype(str)
    label_types = true_labels.apply(classify_label_type)
    noncbl_count = (true_labels == DEFAULT_NONCBL_LABEL).sum()
    logger.info("Label-type breakdown (n=%d):", len(merged))
    logger.info(
        "  serotype:       %d", (label_types == "serotype").sum() - noncbl_count
    )
    logger.info("  serogroup_only: %d", (label_types == "serogroup_only").sum())
    logger.info("  compound:       %d", (label_types == "compound").sum())
    logger.info("  NON-CBL:        %d", noncbl_count)

    merged_path = os.path.join(args.output_dir, "merged_query_results.csv")
    merged.to_csv(merged_path, index=False)
    logger.info("Merged results saved to: %s", merged_path)

    # ── Serotype evaluation: exclude invalid label types ─────
    serotype_exclude = {DEFAULT_NONCBL_LABEL} | set(SEROGROUP_LABELS)
    compound_labels = {
        lbl for lbl in true_labels.unique() if classify_label_type(lbl) == "compound"
    }
    serotype_exclude |= compound_labels

    _evaluate(
        df=merged,
        true_col="true_serotype",
        pred_col="pred_serotype",
        output_dir=args.output_dir,
        report_stem="serotype",
        missing_label=args.missing_label,
        exclude_labels=serotype_exclude,
    )

    # ── Genogroup evaluation: derive truth from biology ──────
    merged["true_genogroup"] = merged["true_serotype"].apply(
        lambda s: map_serotype_to_group(str(s))
    )
    genogroup_pred_col = args.genogroup_column
    if genogroup_pred_col not in merged.columns:
        # Fall back: map predicted serotype through biology mapping
        merged["pred_genogroup"] = merged["pred_serotype"].apply(
            lambda s: map_serotype_to_group(str(s))
        )
        genogroup_pred_col = "pred_genogroup"

    genogroup_exclude = {DEFAULT_NONCBL_LABEL}
    _evaluate(
        df=merged,
        true_col="true_genogroup",
        pred_col=genogroup_pred_col,
        output_dir=args.output_dir,
        report_stem="genogroup",
        missing_label=args.missing_label,
        exclude_labels=genogroup_exclude,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate serotype performance from query results"
    )
    parser.add_argument(
        "--query_results", required=True, help="Path to query_results.csv"
    )
    parser.add_argument(
        "--metadata", required=True, help="Path to metadata CSV/TSV with true labels"
    )
    parser.add_argument("--output_dir", required=True, help="Directory to save reports")

    parser.add_argument(
        "--label_column", default=DEFAULT_LABEL_COLUMN, help="True serotype column name"
    )
    parser.add_argument(
        "--pred_column",
        default=DEFAULT_PRED_COLUMN,
        help="Predicted serotype column name",
    )
    parser.add_argument(
        "--missing_label",
        default=DEFAULT_MISSING_LABEL,
        help="Label representing missing class",
    )

    parser.add_argument(
        "--id_column", default=DEFAULT_ID_COLUMN, help="Metadata ID column"
    )
    parser.add_argument(
        "--contig_column", default=DEFAULT_CONTIG_COLUMN, help="Metadata contig column"
    )
    parser.add_argument(
        "--id_sep", default=CONTIG_SEP, help="Separator used for FASTA record IDs"
    )
    parser.add_argument(
        "--metadata_id_column", default=None, help="Optional direct metadata ID column"
    )

    parser.add_argument(
        "--genogroup_column",
        default=DEFAULT_GENOGROUP_COLUMN,
        help="Genogroup column name",
    )

    main(parser.parse_args())
