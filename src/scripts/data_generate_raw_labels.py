"""
This script generates raw labels from FASTA and metadata files.
It extracts public names from FASTA headers and merges with metadata to create raw labels.
"""

import pandas as pd
from Bio import SeqIO
import argparse
import sys
from pathlib import Path


def extract_public_names_from_fasta(fasta_path):
    """
    Extract public names from FASTA file headers.
    
    Args:
        fasta_path: Path to FASTA file
        
    Returns:
        pandas.Series of public names
    """
    try:
        print(f"Reading FASTA file: {fasta_path}")
        with open(fasta_path) as handle:
            fasta_sequences = SeqIO.parse(handle, 'fasta')
            fasta_ids = [record.id for record in fasta_sequences]
        
        print(f"Found {len(fasta_ids)} sequences in FASTA file")
        
        public_names = pd.Series(
            list(map(lambda rec_id: rec_id.split("__")[0], fasta_ids)), 
            name="Public_name"
        )
        return public_names
    except Exception as e:
        print(f"Error reading FASTA file {fasta_path}: {e}")
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
        print(f"Loading metadata from: {meta_path}")
        metadata = pd.read_csv(meta_path)
        cols = ["Public_name", "Lane_id", "In_silico_serotype"]
        
        # Check if required columns exist
        missing_cols = [col for col in cols if col not in metadata.columns]
        if missing_cols:
            print(f"Error: Missing required columns in metadata: {missing_cols}")
            print(f"Available columns: {list(metadata.columns)}")
            sys.exit(1)
        
        metadata = metadata[cols]
        print(f"Loaded {len(metadata)} metadata entries")
        return metadata
    except Exception as e:
        print(f"Error reading metadata file {meta_path}: {e}")
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
    print("Loading metadata...")
    metadata = load_metadata(meta_path)
    
    print("Extracting public names from FASTA...")
    public_names = extract_public_names_from_fasta(fasta_path)
    
    print("Merging data...")
    labels = pd.DataFrame(public_names).merge(metadata, how="left", on="Public_name") \
        .rename({"In_silico_serotype": "Serotype"}, axis=1)
    
    print(f"Created {len(labels)} label entries")
    
    # Check for missing matches
    missing_matches = labels[labels['Serotype'].isna()]
    if len(missing_matches) > 0:
        print(f"Warning: {len(missing_matches)} sequences from FASTA have no metadata match")
        print("First few missing matches:")
        print(missing_matches['Public_name'].head())
    
    print(f"Saving raw labels to {output_path}")
    labels.to_csv(output_path, sep="\t", index=False)
    
    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Generate raw labels from FASTA and metadata files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python generate_raw_labels.py --fasta data.fasta --metadata meta.csv --output raw_labels.tsv
        """
    )
    
    parser.add_argument("--fasta", required=True, help="Path to FASTA file")
    parser.add_argument("--metadata", required=True, help="Path to metadata CSV file")
    parser.add_argument("--output", required=True, help="Path for output raw labels file")
    
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
    
    print("\nSummary of raw serotypes:")
    serotype_counts = result.Serotype.value_counts()
    print(serotype_counts.head(10))
    print(f"\nTotal entries: {len(result)}")
    print(f"Unique serotypes: {result.Serotype.nunique()}")
    print("Raw labels generation completed successfully!")


if __name__ == "__main__":
    main()
