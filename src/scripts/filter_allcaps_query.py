import argparse
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Filter query results based on score threshold.")
    parser.add_argument("--embeddings_path", type=str, required=True, help="Path to the input NPZ file containing query embeddings.")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to the input CSV file containing samples metadata.")
    parser.add_argument('--serotypes_list', type=str, default=None, help='Comma-separated list of serotypes to include in the plot.')
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output files.")
    parser.add_argument("--output_postfix", type=str, required=True, help="Postfix to append to the output file name.")
    args = parser.parse_args()

    if args.serotypes_list:
        args.serotypes_list = list(map(str.strip, args.serotypes_list.split(',')))
    else:
        args.serotypes_list = None
    
    return args


def main(args):
    embeddings_npz = np.load(args.embeddings_path, allow_pickle=True)
    embeddings, seq_ids = embeddings_npz['embeddings'], embeddings_npz['record_ids']
    record_ids = pd.DataFrame({
        "RecordID": seq_ids  # list(map(lambda x: x[:x.find("#")], seq_ids))
        # "SeqID": seq_ids,
    })
    print(f"Loaded {len(embeddings)} embeddings and {len(record_ids)} record IDs.")
    labels_df = pd.read_csv(args.metadata_path, index_col=0)
    labels_df["RecordID"] = labels_df.index + "#" + labels_df["Contig_ID"].astype(str)
    n_dupes = labels_df["RecordID"].duplicated().sum()
    if n_dupes:
        print(f"Warning: dropping {n_dupes} duplicate RecordID(s) from metadata to keep alignment with embeddings.")
        labels_df = labels_df.drop_duplicates(subset="RecordID")
    # left merge on the embeddings' record_ids so labels_df stays row-aligned with `embeddings`
    labels_df = record_ids.merge(labels_df, on="RecordID", how="left")
    if args.serotypes_list:
        filtered_serotypes = set(args.serotypes_list).intersection(set(labels_df["Serotype"].unique()))
        print(f"Filtering out samples to keep only the following serotypes: {filtered_serotypes}")
        
        mask = labels_df["Serotype"].isin(filtered_serotypes)
        labels_df = labels_df[mask]
        embeddings = embeddings[mask]
    else:
        print("Not filtering out any class from the samples")

    output_npz_path = f"{args.output_dir}/filtered_embeddings_{args.output_postfix}.npz"
    np.savez_compressed(output_npz_path, embeddings=embeddings, record_ids=labels_df["RecordID"].values)
    print(f"Saved filtered embeddings to {output_npz_path}")
    output_csv_path = f"{args.output_dir}/filtered_labels_{args.output_postfix}.csv"
    labels_df.to_csv(output_csv_path, index=False)
    print(f"Saved filtered labels to {output_csv_path}")


if __name__ == "__main__":
    main(parse_args())
