"""
This script loads cleaned serotype labels and integrates them with contig IDs and capsule presence.
It takes a cleaned labels TSV file and outputs a final metadata file with serotype and contig IDs.
"""

import os
import pandas as pd
import argparse
from pathlib import Path

from .consts import DEFAULT_NONCBL_LABEL, CONTIG_SEP
from .data_labels_preprocessing import preprocess_metadata


def main():
    parser = argparse.ArgumentParser(
        description="Incorporate cleaned serotype labels and contig IDs into final metadata",
    )
    parser.add_argument("--clean_labels", required=True, help="Path to cleaned labels TSV file")
    parser.add_argument("--skip_labels", type=str, default="", help="Comma-separated list of labels to skip")
    parser.add_argument("--output_dir", required=True, help="Path for output files")
    parser.add_argument("--label_column", type=str, default="ERR", help="Column name for sample IDs in labels file")

    args = parser.parse_args()
    try:
        args.skip_labels = [label.strip() for label in args.skip_labels.split(",") if label.strip()]
    except ValueError:
        print("Error parsing skip_labels. It should be a comma-separated list of labels. Proceeding with no skips.")
        args.skip_labels = []

    labels = pd.read_csv(args.clean_labels)
    print(f"Loaded {len(labels)} cleaned label entries")
    labels = preprocess_metadata(
        labels,
        skip_labels=args.skip_labels or None,
    )
    print(f"After preprocessing: {len(labels)} entries, {labels.Serotype.nunique()} serotypes")

    # Create output directory if it doesn't exist
    output_path = Path(args.output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = [f.split(".")[0] for f in os.listdir(output_path.parent / "base_embeddings_chunked") if f.endswith(".npy")]
    assert names, "The base_embeddings_chunked directory is empty"
    labels["file_name"] = labels["Public_ID"].astype(str) + CONTIG_SEP + labels["Contig_ID"].astype(str)
    missing = set(names) - set(labels["file_name"])
    print(f"Some embedding files ({len(missing)}) do not have matching metadata entries: {missing}. Removing these from final metadata.")
    labels = labels[labels["file_name"].isin(names)]
    labels = labels.drop(columns=["file_name"])

    output_file = output_path / "final_metadata.csv"
    labels.to_csv(output_file, index=False)
    print(f"Final metadata with serotypes and contig IDs saved to {output_file}")
    print(f"Total entries in final metadata: {len(labels)}")
    print(f"Unique serotypes in final metadata: {labels.Serotype.nunique()}")


if __name__ == "__main__":
    main()
