"""
This script loads raw labels and cleans/standardizes serotype entries.
It takes a raw labels TSV file and outputs a cleaned version with standardized serotypes.
"""

# Example usage: python src/scripts/labels_preprocessing.py --metadata data/monocle-metadata.tsv --output_dir results/
import re
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

from .consts import NON_TYPEABLE, DEFAULT_LABEL_COLUMN


def read_monocle_metadata(file_path, label_column=DEFAULT_LABEL_COLUMN):
    """
    Reads the monocle metadata TSV file into a pandas DataFrame.

    Args:
        file_path: Path to the TSV file
        label_column: Name of the column to use for serotype labels

    Returns:
        pandas DataFrame with metadata
    """
    selected_columns = ["Sample_name", "ERR", "In_silico_serotype"]
    df = pd.read_csv(file_path)[selected_columns]
    df = df.rename(columns={"In_silico_serotype": label_column})
    return df


def cleanup_serotype(value):
    """
    Clean and standardize serotype entries.

    Args:
        value: Raw serotype value from metadata

    Returns:
        Cleaned serotype string
    """

    serotype_mapping = {
        '6A(6A-I)': "6A",
        '6A(6A-II)': "6A",
        '6A(6A-III)': "6A",
        '6A(6A-IV)': "6A",
        '6B(6B-I)': "6B",
        '6E(6B)': "6E",
        '6E(6A)': "6E",
        '11A(11F_LIKE)': "11A",
        '11A/11B/11C/11D/11E/11F/11F_LIKE': "11",
        '19A(19A-I/19A-II)': "19A",
        '19A(19A-I)': "19A",
        '19A(19A-II)': "19A",
        '19F(19AF)': "19F",
        '19F(19F-II)': "19F",
        '19F(19F-III)': "19F",
        '23B(23B1)': "23B",
        '23B1': "23B",
        '24B/24C/24F': "Serogroup 24",
        '33A/33E/33F': "Serogroup 33",
        '33F(33F-1B)': "33F",
        '33F(33F-1B)': "33F",
        'POSSIBLE 6A': "6A",
        'POSSIBLE 6C': "6C",
        'POSSIBLE 6D': "6D",
        'POSSIBLE 6E': "6E",
        "SEROGROUP 24": "Serogroup 24",
        "24": "Serogroup 24",  # <-- TODO rename "24"s to "24F"?
        "33": "Serogroup 33",  # <-- TODO rename "33"s to "33F"?
        # "24A": "Serogroup 24",
        # "24F": "Serogroup 24",
        # '33A/33F': "?",
        # '35B/35D': "?",
    }

    untypables = [
        "NT",
        "SWISS_NT",
        "UNTYPABLE",
        "COVERAGE TOO LOW",
        "ALTERNATIVE_ALIB_NT",
        "SEROBA FAILURE",
        "NCC2_ALIC_ALID_NON_ENCAPSULATED",
        "Non-typeable/NCC2a",
    ]

    if pd.isna(value):
        return NON_TYPEABLE

    # Standardize mappings
    if value in serotype_mapping:
        return serotype_mapping[value]

    if value in untypables:
        return NON_TYPEABLE

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
        print(f"Warning: Unrecognized serotype format '{value}', marking as Non-typeable")
        return NON_TYPEABLE

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
    raw_df = read_monocle_metadata(raw_path)
    print(f"Loaded {len(raw_df)} raw label entries")

    original_serotypes = raw_df.Serotype.unique().tolist()
    print(f"Found {len(original_serotypes)} unique serotypes")

    print("Cleaning serotypes...")
    raw_df["Serotype"] = raw_df["Serotype"].apply(cleanup_serotype)

    print("Dropping Non-typeables...")
    raw_df = raw_df[raw_df["Serotype"] != NON_TYPEABLE]

    cleaned_serotypes = raw_df.Serotype.unique().tolist()
    print(f"After cleaning: {len(cleaned_serotypes)} unique serotypes")
    print(f"The removed serotypes: {set(original_serotypes) - set(cleaned_serotypes)}")
    print(f"Saving cleaned labels to {clean_path}")
    raw_df.to_csv(clean_path, index=False)

    return raw_df


def main():
    parser = argparse.ArgumentParser(
        description="Clean and standardize GPS metadata and serotype labels from raw labels file",
    )

    parser.add_argument("--metadata", required=True, help="Path to raw labels TSV file")
    parser.add_argument("--output_dir", required=True, help="Path for output files")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_path = Path(args.output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process labels
    print("Processing raw labels...")
    result = process_labels(args.metadata, output_path / "cleaned_labels.csv")

    print("\nSummary of cleaned serotypes:")
    serotype_counts = result.Serotype.value_counts().to_dict()
    for serotype in sorted(serotype_counts.keys()):
        print(f"\t- {serotype}:\t {serotype_counts[serotype]}")
    print(f"\nTotal entries: {len(result)}")
    print(f"Unique serotypes: {result.Serotype.nunique()}")
    print("Processing completed successfully!")


if __name__ == "__main__":
    main()
