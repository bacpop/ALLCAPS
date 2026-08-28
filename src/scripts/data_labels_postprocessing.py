"""
Post-process metadata after embedding: keep only samples that have
a matching .npy embedding file, drop the rest.

Expects a *flat* embedding directory (all .npy files in one folder,
no cbl/non-cbl subdirs).  The input metadata should already have clean
labels (via data_labels_preprocessing or data_train_test_split).
"""

import os
import pandas as pd
import argparse
from pathlib import Path

from .consts import CONTIG_SEP
from .data_labels_preprocessing import preprocess_metadata
from .logging_config import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Filter metadata to samples that survived embedding",
    )
    parser.add_argument(
        "--clean_labels", required=True, help="Path to pre-processed metadata CSV/TSV"
    )
    parser.add_argument(
        "--skip_labels",
        type=str,
        default="",
        help="Comma-separated list of labels to skip",
    )
    parser.add_argument(
        "--embedding_dir",
        required=True,
        help="Path to flat directory of .npy embedding files",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to write final_metadata.csv"
    )

    args = parser.parse_args()
    try:
        args.skip_labels = [
            label.strip() for label in args.skip_labels.split(",") if label.strip()
        ]
    except ValueError:
        logger.error(
            "Error parsing skip_labels. It should be a comma-separated list of labels. Proceeding with no skips."
        )
        args.skip_labels = []

    labels = pd.read_csv(
        args.clean_labels,
        sep="\t" if args.clean_labels.endswith(".tsv") else ",",
    )
    labels = preprocess_metadata(  # The cleaning part is redundant
        labels,
        skip_labels=args.skip_labels or None,
    )
    logger.info(
        "After preprocessing: %d entries, %d serotypes",
        len(labels),
        labels.Serotype.nunique(),
    )
    logger.info("Loaded %d metadata entries", len(labels))

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    embedding_dir = Path(args.embedding_dir)
    names = [f.split(".")[0] for f in os.listdir(embedding_dir) if f.endswith(".npy")]
    assert names, f"No .npy files found in {embedding_dir}"

    labels["file_name"] = (
        labels["Public_ID"].astype(str) + CONTIG_SEP + labels["Contig_ID"].astype(str)
    )

    embedded = set(names)
    in_meta = set(labels["file_name"])
    missing_embeddings = in_meta - embedded
    missing_metadata = embedded - in_meta
    if missing_embeddings:
        logger.warning(
            "%d metadata entries have no embedding file (dropped)",
            len(missing_embeddings),
        )
    if missing_metadata:
        logger.warning(
            "%d embedding files have no metadata entry (ignored)", len(missing_metadata)
        )

    pre_count = labels.copy()
    labels = labels[labels["file_name"].isin(embedded)]
    labels = labels.drop(columns=["file_name"])

    # ── Per-serotype retention audit ─────────────────────────
    pre_caps = pre_count[pre_count["Is_capsule"] == 1]
    post_caps = labels[labels["Is_capsule"] == 1]
    before = pre_caps.groupby("Serotype").size()
    after = post_caps.groupby("Serotype").size().reindex(before.index, fill_value=0)
    retention = (after / before).fillna(0.0)
    audit = (
        pd.DataFrame(
            {"before": before, "after": after, "retention": retention.round(4)}
        )
        .sort_values("retention")
    )
    audit_path = output_path / "embedding_retention_audit.csv"
    audit.to_csv(audit_path)
    logger.info("Per-serotype retention audit saved to %s", audit_path)

    overall = len(post_caps) / max(len(pre_caps), 1)
    logger.info(
        "CBL retention: %d/%d (%.1f%%) overall after embedding",
        len(post_caps),
        len(pre_caps),
        100 * overall,
    )
    bad = audit[(audit["retention"] < 0.8) & (audit["before"] >= 5)]
    if not bad.empty:
        logger.warning(
            "Serotypes with <80%% retention (likely under-trained, biased predictions on test):\n%s",
            bad.head(20).to_string(),
        )
        if (bad["retention"] < 0.5).any():
            logger.error(
                "%d serotypes lost >50%% of samples during embedding. "
                "Train/test class distributions are now skewed — fix embed_transformer.py "
                "before training.",
                int((bad["retention"] < 0.5).sum()),
            )

    output_file = output_path / "final_metadata.csv"
    labels.to_csv(output_file, index=False)
    logger.info("Final metadata saved to %s", output_file)
    logger.info("%d entries, %d serotypes", len(labels), labels.Serotype.nunique())


if __name__ == "__main__":
    main()
