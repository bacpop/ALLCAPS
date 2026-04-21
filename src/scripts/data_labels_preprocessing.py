"""
This script loads raw labels and cleans/standardizes serotype entries.
It takes a raw labels TSV file and outputs a cleaned version with standardized serotypes.
"""

# Example usage: python src/scripts/labels_preprocessing.py --metadata data/monocle-metadata.tsv --output_dir results/
import os
import re
import pandas as pd
import argparse
from pathlib import Path
from Bio import SeqIO

from .consts import (
    NON_TYPEABLE,
    DEFAULT_LABEL_COLUMN,
    DEFAULT_ID_COLUMN,
    DEFAULT_CONTIG_COLUMN,
    DEFAULT_NONCBL_LABEL,
    CONTIG_SEP,
)
from .logging_config import get_logger

logger = get_logger(__name__)


def read_monocle_metadata(
    file_path, label_column=DEFAULT_LABEL_COLUMN, id_column=DEFAULT_ID_COLUMN
):
    """
    Reads the monocle metadata TSV file into a pandas DataFrame.

    Args:
        file_path: Path to the TSV file
        label_column: Name of the column to use for serotype labels

    Returns:
        pandas DataFrame with metadata
    """
    selected_columns = ["ERR", "In_silico_serotype"]
    df = pd.read_csv(file_path)[selected_columns]
    df = df.rename(columns={"In_silico_serotype": label_column, "ERR": id_column})
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
        "6A(6A-I)": "6A",
        "6A(6A-II)": "6A",
        "6A(6A-III)": "6A",
        "6A(6A-IV)": "6A",
        "6B(6B-I)": "6B",
        "6E(6B)": "6E",
        "6E(6A)": "6E",
        "11A(11F_LIKE)": "11A",
        "11A/11B/11C/11D/11E/11F/11F_LIKE": "11",
        "19A(19A-I/19A-II)": "19A",
        "19A(19A-I)": "19A",
        "19A(19A-II)": "19A",
        "19F(19AF)": "19F",
        "19F(19F-II)": "19F",
        "19F(19F-III)": "19F",
        "23B(23B1)": "23B",
        "23B1": "23B",
        "24B/24C/24F": "Serogroup 24",
        "33A/33E/33F": "Serogroup 33",
        "33F(33F-1A)": "33F",
        "33F(33F-1B)": "33F",
        "POSSIBLE 6A": "6A",
        "POSSIBLE 6C": "6C",
        "POSSIBLE 6D": "6D",
        "POSSIBLE 6E": "6E",
        "SEROGROUP 24": "Serogroup 24",
        "SEROGROUP 33": "Serogroup 33",
        "24": "Serogroup 24",
        "33": "Serogroup 33",
        # "24A": "Serogroup 24",
        # "24F": "Serogroup 24",
        # '33A/33F': "?",
        # '35B/35D': "?",
    }

    WHITELIST = ["NON-CBL"]

    untypables = [
        "NON-TYPEABLE",
        "NT",
        "NTR",
        "SWISS_NT",
        "UNTYPABLE",
        "COVERAGE TOO LOW",
        "ALTERNATIVE_ALIB_NT",
        "SEROBA FAILURE",
        "NCC2_ALIC_ALID_NON_ENCAPSULATED",
        "NON-TYPEABLE/NCC2A",
        "NOVEL PATTERN",
    ]

    if pd.isna(value):
        return NON_TYPEABLE
    value = str(value).strip().upper()

    # Standardize mappings
    if value in serotype_mapping:
        return serotype_mapping[value]

    if value in untypables:
        return NON_TYPEABLE

    if value in WHITELIST:
        return value

    # Replace incorrect formats like "06B" with "6B"
    value = re.sub(r"^0+(\d+)", r"\1", value)

    # Handle specific typos and corrections
    typo_corrections = {
        "18C/19F": ["18C/19F"],
        "15B/C": ["15B/15C"],
        # Add more corrections as needed
    }

    for correct_form, variants in typo_corrections.items():
        if value in variants:
            return correct_form

    # If still irrelevant, mark as Non-typeable
    if not re.match(r"^[A-Z0-9/]+$", value):
        logger.warning(
            "Unrecognized serotype format '%s', marking as Non-typeable", value
        )
        return NON_TYPEABLE

    return value


def preprocess_metadata(
    df: pd.DataFrame,
    serotype_column: str = DEFAULT_LABEL_COLUMN,
    drop_nontypeable: bool = True,
    skip_labels: list = None,
) -> pd.DataFrame:
    """Unified metadata preprocessing: clean serotypes, optionally drop
    non-typeables and specific labels.

    This is the single entry-point for all label cleaning so that
    training and evaluation always see identical label sets.
    """
    df = df.copy()
    df[serotype_column] = df[serotype_column].map(cleanup_serotype)
    if drop_nontypeable:
        df = df[df[serotype_column] != NON_TYPEABLE]
    if skip_labels:
        df = df[~df[serotype_column].isin(skip_labels)]
    return df


def process_labels(raw_path, clean_path):
    """
    Process existing raw labels file and clean serotypes.

    Args:
        raw_path: Path to raw labels TSV file
        clean_path: Path for output cleaned labels file

    Returns:
        Cleaned pandas DataFrame
    """
    logger.info("Loading raw labels from %s", raw_path)
    raw_df = read_monocle_metadata(raw_path)
    logger.info("Loaded %d raw label entries", len(raw_df))

    original_serotypes = raw_df.Serotype.unique().tolist()
    logger.info("Found %d unique serotypes", len(original_serotypes))

    logger.info("Cleaning serotypes...")
    raw_df = preprocess_metadata(raw_df)

    cleaned_serotypes = raw_df.Serotype.unique().tolist()
    logger.info("After cleaning: %d unique serotypes", len(cleaned_serotypes))
    logger.info(
        "The removed serotypes: %s", set(original_serotypes) - set(cleaned_serotypes)
    )
    logger.info("Saving cleaned labels to %s", clean_path)
    raw_df.to_csv(clean_path, index=False)

    return raw_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean and standardize GPS metadata and serotype labels from raw labels file",
    )

    parser.add_argument("--metadata", required=True, help="Path to raw labels TSV file")
    parser.add_argument("--output_dir", required=True, help="Path for output files")
    parser.add_argument(
        "--cbl-fasta",
        required=True,
        help="Path to FASTA file containing CBL contigs for appending to metadata",
    )
    parser.add_argument(
        "--noncbl-fasta",
        required=True,
        help="Path to FASTA file containing non-CBL contigs for appending to metadata",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.metadata):
        raise FileNotFoundError(f"Metadata file not found: {args.metadata}")
    if not os.path.isfile(args.cbl_fasta):
        raise FileNotFoundError(f"CBL FASTA file not found: {args.cbl_fasta}")
    if not os.path.isfile(args.noncbl_fasta):
        raise FileNotFoundError(f"Non-CBL FASTA file not found: {args.noncbl_fasta}")

    return args


def read_fasta_ids(fasta_path):
    """Return the list of record IDs from a FASTA file (headers only)."""
    return [record.id for record in SeqIO.parse(fasta_path, "fasta")]


def build_contig_metadata(cbl_fasta, noncbl_fasta, clean_labels):
    """Combine CBL and non-CBL FASTA record IDs with cleaned serotype labels.

    cbl record id format:     ERR1788086#7                 (Public_ID#Contig_ID)
    non-cbl record id format: NONCBL#ERR1788086#7          (NONCBL#Public_ID#Contig_ID)

    Returns a DataFrame with columns: Public_ID, Contig_ID, Serotype, Is_capsule.
    For non-CBLs, the NONCBL# prefix is retained on Public_ID so downstream
    lookups against per-contig artifacts (e.g. embedding files) still match.
    """
    cbl_ids = read_fasta_ids(cbl_fasta)
    noncbl_ids = read_fasta_ids(noncbl_fasta)
    logger.info(
        "Loaded %d CBL and %d non-CBL FASTA records", len(cbl_ids), len(noncbl_ids)
    )

    cbl_parts = [rid.split(CONTIG_SEP, 1) for rid in cbl_ids]
    if any(len(p) != 2 for p in cbl_parts):
        raise ValueError(
            f"CBL record IDs must have format Public_ID{CONTIG_SEP}Contig_ID"
        )
    cbl_df = pd.DataFrame(cbl_parts, columns=[DEFAULT_ID_COLUMN, DEFAULT_CONTIG_COLUMN])
    cbl_df = cbl_df.merge(
        clean_labels[[DEFAULT_ID_COLUMN, DEFAULT_LABEL_COLUMN]].drop_duplicates(),
        on=DEFAULT_ID_COLUMN,
        how="left",
    )
    missing = cbl_df[DEFAULT_LABEL_COLUMN].isna().sum()
    if missing:
        logger.warning("%d CBL contigs dropped (no matching cleaned label)", missing)
        cbl_df = cbl_df.dropna(subset=[DEFAULT_LABEL_COLUMN])
    cbl_df["Is_capsule"] = 1

    noncbl_parts = []
    for rid in noncbl_ids:
        pieces = rid.split(CONTIG_SEP)
        if len(pieces) < 3:
            raise ValueError(
                f"Non-CBL record ID '{rid}' must have format NONCBL{CONTIG_SEP}Public_ID{CONTIG_SEP}Contig_ID"
            )
        prefix, public_id, contig_id = (
            pieces[0],
            CONTIG_SEP.join(pieces[1:-1]),
            pieces[-1],
        )
        noncbl_parts.append((f"{prefix}{CONTIG_SEP}{public_id}", contig_id))
    noncbl_df = pd.DataFrame(
        noncbl_parts, columns=[DEFAULT_ID_COLUMN, DEFAULT_CONTIG_COLUMN]
    )
    noncbl_df[DEFAULT_LABEL_COLUMN] = DEFAULT_NONCBL_LABEL
    noncbl_df["Is_capsule"] = 0

    columns = [
        DEFAULT_ID_COLUMN,
        DEFAULT_CONTIG_COLUMN,
        DEFAULT_LABEL_COLUMN,
        "Is_capsule",
    ]
    return pd.concat([cbl_df[columns], noncbl_df[columns]], ignore_index=True)


def main(args):
    # Create output directory if it doesn't exist
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process labels
    logger.info("Processing raw labels...")
    result = process_labels(args.metadata, output_path / "cleaned_labels.csv")

    logger.info("Summary of cleaned serotypes:")
    serotype_counts = result.Serotype.value_counts().to_dict()
    for serotype in sorted(serotype_counts.keys()):
        logger.info("\t- %s:\t %d", serotype, serotype_counts[serotype])
    logger.info("Total entries: %d", len(result))
    logger.info("Unique serotypes: %d", result.Serotype.nunique())

    logger.info("Building final contig-level metadata from FASTA record IDs...")
    final_df = build_contig_metadata(args.cbl_fasta, args.noncbl_fasta, result)

    final_file = output_path / "initial_metadata.csv"
    final_df.to_csv(final_file, index=False)
    logger.info("Early clean metadata saved to %s", final_file)
    logger.info(
        "Early metadata: %d entries (%d CBL, %d non-CBL), %d unique serotypes",
        len(final_df),
        int((final_df["Is_capsule"] == 1).sum()),
        int((final_df["Is_capsule"] == 0).sum()),
        final_df[DEFAULT_LABEL_COLUMN].nunique(),
    )

    logger.info("Processing completed successfully!")


if __name__ == "__main__":
    main(parse_args())
