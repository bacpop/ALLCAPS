"""
This script loads cleaned serotype labels and integrates them with contig IDs and capsule presence.
It takes a cleaned labels TSV file and outputs a final metadata file with serotype and contig IDs.
"""

# Example usage: python src/scripts/labels_postprocessing.py --clean_labels data/GPS_All_clean_labels.tsv --output_dir results/
import os
import pandas as pd
import argparse
from pathlib import Path

from consts import DEFAULT_NONCBL_LABEL, CONTIG_SEP


def main():
    parser = argparse.ArgumentParser(
        description="Incorporate cleaned serotype labels and contig IDs into final metadata",
    )

    parser.add_argument(
        "--clean_labels", required=True, help="Path to cleaned labels TSV file"
    )
    parser.add_argument("--output_dir", required=True, help="Path for output files")

    args = parser.parse_args()

    non_cbl_label = DEFAULT_NONCBL_LABEL

    labels = pd.read_csv(args.clean_labels, sep="\t")
    print(f"Loaded {len(labels)} cleaned label entries")
    print(f"Unique serotypes in cleaned labels: {labels.Serotype.nunique()}")

    # Create output directory if it doesn't exist
    output_path = Path(args.output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add capsule presence column and contig id column after saving chunked embeddings
    results = pd.DataFrame(columns=["Public_ID", "Contig_ID", "Serotype", "Is_capsule"])
    for is_capsule, subdir in zip(
        [1, 0], ["base_embeddings_chunked/cbl", "base_embeddings_chunked/non-cbl"]
    ):
        names = [f.split(".")[0] for f in os.listdir(output_path / subdir)]
        names = [name for name in names if name.endswith(".npy")]
        public_ids, contig_ids = zip(*[name.split(CONTIG_SEP) for name in names])
        if is_capsule:
            serotypes = (
                pd.DataFrame(public_ids, columns=["Public_name"])
                .merge(
                    labels[["Public_name", "Serotype"]].drop_duplicates(), how="left"
                )["Serotype"]
                .values
            )
        else:
            serotypes = [non_cbl_label] * len(public_ids)
        results = pd.concat(
            [
                results,
                pd.DataFrame(
                    {
                        "Public_ID": public_ids,
                        "Contig_ID": contig_ids,
                        "Serotype": serotypes,
                        "Is_capsule": [is_capsule] * len(public_ids),
                    }
                ),
            ],
            ignore_index=True,
        )

    output_file = output_path / "final_metadata.tsv"
    results.to_csv(output_file, sep="\t", index=False)
    print(f"Final metadata with serotypes and contig IDs saved to {output_file}")
    print(f"Total entries in final metadata: {len(results)}")
    print(f"Unique serotypes in final metadata: {results.Serotype.nunique()}")


if __name__ == "__main__":
    main()
