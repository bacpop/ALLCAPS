import argparse

import os
import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import map_serotype_to_group

MISSING_LABEL = "Non-typeable"  # TODO: Make this a parameter
DOWNSAMPLE_SIZE = 1000  # TODO: Make this a parameter
LABEL_COLUMN = "serotype"  # TODO: Do sth about it


def calculate_umap(embeddings, labels, output_prefix):
    out_path = output_prefix + ".umap.csv"
    # if os.path.exists(out_path):
    #     print(f"UMAP file already exists: {out_path}")
    #     return pd.read_csv(out_path)
    reducer = umap.UMAP(random_state=42).fit(embeddings)
    UMAP_embedding = reducer.transform(embeddings)
    UMAP_embedding_df = pd.DataFrame(UMAP_embedding)

    UMAP_embedding_df.insert(0, 'Sample', labels.index)
    UMAP_embedding_df.insert(1, 'Serotype', labels.Serotype.tolist())
    UMAP_embedding_df.columns = ['Sample', 'Serotype', 'UMAP1', 'UMAP2']
    UMAP_embedding_df.to_csv(out_path, index=False)
    return UMAP_embedding_df


def plot_umap(df, output_path):
    """
    Plot UMAP visualization of the dataframe.

    Parameters:
    - data: DataFrame containing the data to visualize. The columns are ['UMAP1', 'UMAP2', 'Serotype'].
    - output_path: Path to save the UMAP plot.
    """

    serotypes = df['Serotype'].apply(map_serotype_to_group)
    unique_serotypes = serotypes.unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_serotypes)))  # TODO explore colormaps
    color_map = dict(zip(unique_serotypes, colors))

    plt.figure(figsize=(15, 15))
    for serotype, color in color_map.items():
        subset = df[serotypes == serotype]
        plt.scatter(subset['UMAP1'], subset['UMAP2'], label=serotype, color=color, alpha=0.7, s=10)

    plt.title('UMAP Visualization')
    plt.xlabel('UMAP1'), plt.ylabel('UMAP2')

    plt.legend(title="Serogroups", loc='best', ncol=2, fontsize='small', markerscale=2.0, facecolor='darkgray')
    plt.gcf().patch.set_facecolor('black')
    plt.gca().set_facecolor('black')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="UMAP visualization of data.")
    parser.add_argument('--embeddings', type=str, required=True, help='Path to the embeddings.')
    parser.add_argument('--labels', type=str, required=True, help='Path to the labels file (CSV format).')
    parser.add_argument('--output', type=str, required=True, help='Path to save the UMAP plot.')
    parser.add_argument('--title', type=str, default="UMAP", help='Title of the plot.')
    parser.add_argument('--show-untypable', action='store_true', default=False, help='Include data with missing label in the plot.')
    parser.add_argument('--downsample', action='store_true', default=False, help='Downsample the data for faster plotting.')
    return parser.parse_args()


def main(args):
    print("Loading the data...")
    embeddings = np.load(args.embeddings)
    labels = pd.read_csv(args.labels, sep="\t", index_col=0)
    is_emb_npz = isinstance(embeddings, np.lib.npyio.NpzFile)
    if not is_emb_npz:
        assert embeddings.shape[0] == labels.shape[0], "Number of embeddings and labels do not match."

    assert embeddings.shape[0] == labels.shape[0], "Number of embeddings and labels do not match."

    labels = labels \
        .rename({LABEL_COLUMN: "Serotype"}, axis=1) \
        .fillna(MISSING_LABEL)
    indices_mask = np.ones(len(labels), dtype=bool) if args.show_untypable else labels["Serotype"] != MISSING_LABEL

    if args.downsample:
        print("Downsampling the data for faster plotting...")
        downsample_indices = np.random.choice(np.where(indices_mask)[0], size=DOWNSAMPLE_SIZE, replace=False)
        indices_mask = np.zeros(len(labels), dtype=bool)
        indices_mask[downsample_indices] = True

    print("Calculating UMAP...")
    X = np.array([embeddings[key] for key in labels[indices_mask]["Public_name"]]) if is_emb_npz else embeddings[indices_mask]
    UMAP_embedding_df = calculate_umap(X, labels[indices_mask], args.output)
    print("Plotting UMAP...")
    plot_umap(UMAP_embedding_df, args.output)


if __name__ == "__main__":
    args = parse_args()
    main(args)
