"""
This script loads raw labels and cleans/standardizes serotype entries.
It takes a raw labels TSV file and outputs a cleaned version with standardized serotypes.
"""

# Example usage: python src/scripts/labels_preprocessing.py --raw_labels data/GPS_All_raw_labels.tsv --output_dir results/
import os
import re
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

from consts import DEFAULT_NONCBL_LABEL, CONTIG_SEP


def cleanup_serotype(value):
    """
    Clean and standardize serotype entries.

    Args:
        value: Raw serotype value from metadata

    Returns:
        Cleaned serotype string
    """
    serotype_mapping = {
        "NT": "Non-typeable",
        "SWISS_NT": "Non-typeable",
        "UNTYPABLE": "Non-typeable",
        "COVERAGE TOO LOW": "Non-typeable",
        "ALTERNATIVE_ALIB_NT": "Non-typeable",
        "Non-typeable/NCC2a": "Non-typeable",
        "SEROBA FAILURE": "Non-typeable",
        "SEROGROUP 24": "24",
    }

    untypables = [
        "NT",
        "SWISS_NT",
        "UNTYPABLE",
        "COVERAGE TOO LOW",
        "ALTERNATIVE_ALIB_NT",
        "SEROBA FAILURE",
    ]

    if pd.isna(value):
        return np.nan

    # Standardize mappings
    if value in serotype_mapping:
        return serotype_mapping[value]

    if value in untypables:
        return "Non-typeable"

    value = value.strip().upper()

    # Replace incorrect formats like "06B" with "6B"
    value = re.sub(r"^0+(\d+)", r"\1", value)

    # Handle specific typos and corrections
    typo_corrections = {
        "18C/19F": ["18C/19F"],
        # Add more corrections as needed
    }

    for correct_form, variants in typo_corrections.items():
        if value in variants:
            return correct_form

    # If still irrelevant, mark as Non-typeable
    if not re.match(r"^[A-Z0-9/]+$", value):
        return "Non-typeable"

    return value


def process_labels(raw_path, clean_path):
    """
    Process existing raw labels file and clean serotypes.

    Args:
        raw_path: Path to raw labels TSV file
        clean_path: Path for output cleaned labels file

    Returns:
        Cleaned pandas DataFrame
    """
    print(f"Loading raw labels from {raw_path}")
    raw_df = pd.read_csv(raw_path, sep="\t")
    print(f"Loaded {len(raw_df)} raw label entries")

    original_serotypes = raw_df.Serotype.unique().tolist()
    print(f"Found {len(original_serotypes)} unique serotypes")

    print("Cleaning serotypes...")
    raw_df["Serotype"] = raw_df["Serotype"].apply(cleanup_serotype)

    cleaned_serotypes = raw_df.Serotype.unique().tolist()
    print(f"After cleaning: {len(cleaned_serotypes)} unique serotypes")
    print(f"The removed serotypes: {set(original_serotypes) - set(cleaned_serotypes)}")
    print(f"Saving cleaned labels to {clean_path}")
    raw_df.to_csv(clean_path, sep="\t", index=False)

    return raw_df


def main():
    parser = argparse.ArgumentParser(
        description="Clean and standardize GPS serotype labels from raw labels file",
    )

    parser.add_argument(
        "--raw_labels", required=True, help="Path to raw labels TSV file"
    )
    parser.add_argument("--output_dir", required=True, help="Path for output files")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_path = Path(args.output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process labels
    print("Processing raw labels...")
    result = process_labels(args.raw_labels, output_path / "cleaned_labels.tsv")

    print("\nSummary of cleaned serotypes:")
    serotype_counts = result.Serotype.value_counts()
    print(serotype_counts.head(10))
    print(f"\nTotal entries: {len(result)}")
    print(f"Unique serotypes: {result.Serotype.nunique()}")
    print("Processing completed successfully!")


if __name__ == "__main__":
    main()
