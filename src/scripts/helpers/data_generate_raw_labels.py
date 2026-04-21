"""
This script generates raw labels from FASTA and metadata files.
It extracts public names from FASTA headers and merges with metadata to create raw labels.
"""

import pandas as pd
from Bio import SeqIO
import argparse
import sys
from pathlib import Path

from ..logging_config import get_logger

logger = get_logger(__name__)


def extract_public_names_from_fasta(fasta_path):
    """
    Extract public names from FASTA file headers.

    Args:
        fasta_path: Path to FASTA file

    Returns:
        pandas.Series of public names
    """
    try:
        logger.info("Reading FASTA file: %s", fasta_path)
        with open(fasta_path) as handle:
            fasta_sequences = SeqIO.parse(handle, "fasta")
            fasta_ids = [record.id for record in fasta_sequences]

        logger.info("Found %d sequences in FASTA file", len(fasta_ids))

        public_names = pd.Series(
            list(map(lambda rec_id: rec_id.split("__")[0], fasta_ids)),
            name="Public_name",
        )
        return public_names
    except Exception as e:
        logger.error("Error reading FASTA file %s: %s", fasta_path, e)
        sys.exit(1)


def load_metadata(meta_path):
    """
    Load and filter metadata.

    Args:
        meta_path: Path to metadata CSV file

    Returns:
        Filtered pandas DataFrame
    """
    try:
        logger.info("Loading metadata from: %s", meta_path)
        metadata = pd.read_csv(meta_path)
        cols = ["Public_name", "Lane_id", "In_silico_serotype"]

        # Check if required columns exist
        missing_cols = [col for col in cols if col not in metadata.columns]
        if missing_cols:
            logger.error("Missing required columns in metadata: %s", missing_cols)
            logger.error("Available columns: %s", list(metadata.columns))
            sys.exit(1)

        metadata = metadata[cols]
        logger.info("Loaded %d metadata entries", len(metadata))
        return metadata
    except Exception as e:
        logger.error("Error reading metadata file %s: %s", meta_path, e)
        sys.exit(1)


def generate_raw_labels(fasta_path, meta_path, output_path):
    """
    Generate raw labels from FASTA and metadata files.

    Args:
        fasta_path: Path to FASTA file
        meta_path: Path to metadata CSV file
        output_path: Path for output raw labels file

    Returns:
        pandas.DataFrame with raw labels
    """
    logger.info("Loading metadata...")
    metadata = load_metadata(meta_path)

    logger.info("Extracting public names from FASTA...")
    public_names = extract_public_names_from_fasta(fasta_path)

    logger.info("Merging data...")
    labels = (
        pd.DataFrame(public_names)
        .merge(metadata, how="left", on="Public_name")
        .rename({"In_silico_serotype": "Serotype"}, axis=1)
    )

    logger.info("Created %d label entries", len(labels))

    # Check for missing matches
    missing_matches = labels[labels["Serotype"].isna()]
    if len(missing_matches) > 0:
        logger.warning(
            "%d sequences from FASTA have no metadata match", len(missing_matches)
        )
        logger.warning(
            "First few missing matches: %s",
            missing_matches["Public_name"].head().tolist(),
        )

    logger.info("Saving raw labels to %s", output_path)
    labels.to_csv(output_path, sep="\t", index=False)

    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Generate raw labels from FASTA and metadata files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python generate_raw_labels.py --fasta data.fasta --metadata meta.csv --output raw_labels.tsv
        """,
    )

    parser.add_argument("--fasta", required=True, help="Path to FASTA file")
    parser.add_argument("--metadata", required=True, help="Path to metadata CSV file")
    parser.add_argument(
        "--output", required=True, help="Path for output raw labels file"
    )

    args = parser.parse_args()

    # Validate input files exist
    if not Path(args.fasta).exists():
        raise FileNotFoundError(f"FASTA file {args.fasta} does not exist.")
    if not Path(args.metadata).exists():
        raise FileNotFoundError(f"Metadata file {args.metadata} does not exist.")

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate raw labels
    result = generate_raw_labels(args.fasta, args.metadata, args.output)

    logger.info(
        "Summary of raw serotypes:\n%s", result.Serotype.value_counts().head(10)
    )
    logger.info("Total entries: %d", len(result))
    logger.info("Unique serotypes: %d", result.Serotype.nunique())
    logger.info("Raw labels generation completed successfully!")


if __name__ == "__main__":
    main()
