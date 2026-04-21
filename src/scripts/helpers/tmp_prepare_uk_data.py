"""Helper script to create a proper metadata file and fasta files for the UK dataset."""

import os
import argparse

from Bio import SeqIO
import pandas as pd

from ..consts import CONTIG_SEP
from ..data_labels_preprocessing import preprocess_metadata
from ..logging_config import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare UK dataset metadata and FASTA files"
    )
    parser.add_argument(
        "--metadata", required=True, help="Path to the UK dataset metadata CSV file"
    )
    parser.add_argument(
        "--fasta", required=True, help="Path to the UK dataset FASTA file"
    )
    parser.add_argument(
        "--output_dir", default=".", help="Directory to save processed files"
    )
    parser.add_argument(
        "--output_metadata",
        default="uk_metadata.csv",
        help="Path to save the processed metadata CSV file",
    )
    parser.add_argument(
        "--output_fasta",
        default="uk_sequences.fasta",
        help="Path to save the processed FASTA file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load data
    fasta_records = list(SeqIO.parse(args.fasta, "fasta"))
    record_ids = [record.id for record in fasta_records]
    sample_names = [rid.split("__")[0] for rid in record_ids]
    contig_ids = list(
        map(
            int,
            [
                rid[rid.find("contig") + 6 : rid.find("contig") + 9]
                for rid in record_ids
            ],
        )
    )
    old_record_desc = [record.description for record in fasta_records]
    new_record_desc = [" ".join(desc.split(" ")[2:]) for desc in old_record_desc]
    new_records = []
    for record, sample_name, contig_id, new_record_description in zip(
        fasta_records, sample_names, contig_ids, new_record_desc
    ):
        record.id = f"UK|{sample_name}{CONTIG_SEP}{contig_id}"
        record.description = new_record_description
        new_records.append(record)

    metadata = pd.read_csv(args.metadata)[["ENA", "Agglutination"]].rename(
        columns={"ENA": "Public_ID", "Agglutination": "Serotype"}
    )
    metadata = preprocess_metadata(metadata)
    records_meta = pd.DataFrame(
        {
            "Public_ID": [f"UK|{name}" for name in sample_names],
            # "Sample_name": sample_names,
            "Contig_ID": contig_ids,
            "Is_capsule": [1] * len(sample_names),
        }
    )
    metadata = records_meta.merge(metadata, on="Public_ID", how="left")

    # Save processed metadata
    os.makedirs(args.output_dir, exist_ok=True)
    output_metadata_path = os.path.join(args.output_dir, args.output_metadata)
    metadata.to_csv(output_metadata_path, index=False)
    print(f"Saved processed metadata to {output_metadata_path}")

    # Save processed FASTA
    output_fasta_path = os.path.join(args.output_dir, args.output_fasta)
    SeqIO.write(new_records, output_fasta_path, "fasta")
    print(f"Saved processed FASTA to {output_fasta_path}")


if __name__ == "__main__":
    main()
