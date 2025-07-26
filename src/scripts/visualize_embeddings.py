import json
import argparse
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from utils import map_serotype_to_group
from consts import (
    DEFAULT_MISSING_LABEL, DEFAULT_LABEL_COLUMN,
    DEFAULT_DOWNSAMPLE_SIZE, DEFAULT_SEP
)


def calculate_pca(embeddings, labels, output_prefix):
    out_path = output_prefix + ".pca.csv"
    pca = PCA(n_components=2, random_state=42).fit(embeddings)
    PCA_embedding = pca.transform(embeddings)
    PCA_embedding_df = pd.DataFrame(PCA_embedding)

    PCA_embedding_df.insert(0, 'Sample', labels.index)
    PCA_embedding_df.insert(1, 'Serotype', labels.Serotype.tolist())
    PCA_embedding_df.columns = ['Sample', 'Serotype', 'PCA1', 'PCA2']
    PCA_embedding_df.to_csv(out_path, index=False)
    return pca, PCA_embedding_df

def calculate_tsne(embeddings, labels, output_prefix):
    out_path = output_prefix + ".tsne.csv"
    tsne = TSNE(n_components=2, random_state=42, n_iter=1000)
    TSNE_embedding = tsne.fit_transform(embeddings)
    TSNE_embedding_df = pd.DataFrame(TSNE_embedding)

    TSNE_embedding_df.insert(0, 'Sample', labels.index)
    TSNE_embedding_df.insert(1, 'Serotype', labels.Serotype.tolist())
    TSNE_embedding_df.columns = ['Sample', 'Serotype', 'TSNE1', 'TSNE2']
    TSNE_embedding_df.to_csv(out_path, index=False)
    return tsne, TSNE_embedding_df


def calculate_umap(embeddings, labels, output_prefix):
    out_path = output_prefix + ".umap.csv"
    reducer = umap.UMAP(random_state=42).fit(embeddings)
    UMAP_embedding = reducer.transform(embeddings)
    UMAP_embedding_df = pd.DataFrame(UMAP_embedding)

    UMAP_embedding_df.insert(0, 'Sample', labels.index)
    UMAP_embedding_df.insert(1, 'Serotype', labels.Serotype.tolist())
    UMAP_embedding_df.columns = ['Sample', 'Serotype', 'UMAP1', 'UMAP2']
    UMAP_embedding_df.to_csv(out_path, index=False)
    return reducer, UMAP_embedding_df


def plot_projection(df, method, output_dir):
    """
    Plot the visualization of the dataframe.

    Parameters:
    - data: DataFrame containing the data to visualize. The columns are ['{method}1', '{method}2', 'Serotype'].
    - output_path: Path to save the UMAP plot.
    """

    serotypes = df['Serotype'].apply(map_serotype_to_group)
    unique_serotypes = serotypes.unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_serotypes)))  # TODO explore colormaps
    color_map = dict(zip(unique_serotypes, colors))

    plt.figure(figsize=(15, 15))
    for serotype, color in color_map.items():
        subset = df[serotypes == serotype]
        plt.scatter(subset[f'{method}1'], subset[f'{method}2'], label=serotype, color=color, alpha=0.7, s=10)

    plt.title(f'{method} Visualization')
    plt.xlabel(f'{method}1'), plt.ylabel(f'{method}2')

    plt.legend(title="Serogroups", loc='best', ncol=2, fontsize='small', markerscale=2.0, facecolor='darkgray')
    plt.gcf().patch.set_facecolor('black')
    plt.gca().set_facecolor('black')

    plt.tight_layout()
    plt.savefig(output_dir + f'/{method.lower()}_visualization.pdf')
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="2D visualization of data.")
    parser.add_argument('--embeddings', type=str, required=True, help='Path to the embeddings.')
    parser.add_argument('--labels', type=str, required=True, help='Path to the labels file (CSV format).')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save the UMAP plot.')
    parser.add_argument('--method', type=str, choices=['umap', 'tsne', 'pca'], default='umap',
                        help='Dimensionality reduction method to use for visualization.')
    parser.add_argument('--params', type=str, default="{}", help='JSON string of parameters.')
    parser.add_argument('--title', type=str, default="UMAP", help='Title of the plot.')
    parser.add_argument('--showـuntypable', action='store_true', default=False, help='Include data with missing label in the plot.')
    parser.add_argument('--downsample', action='store_true', default=False, help='Downsample the data for faster plotting.')
    parser.add_argument('--serotypesـlist', type=str, default=None, help='Comma-separated list of serotypes to include in the plot.')
    args = parser.parse_args()

    try:
        args.params = json.loads(args.params)
        if not isinstance(args.params, dict):
            print("Model parameters should be a JSON object.")
            args.params = {}
    except json.JSONDecodeError:
        print("Error parsing model parameters JSON string.")
        args.params = {}
    finally:
        print("Model parameters:", args.params)

    if args.serotypes_list:
        args.serotypes_list = list(map(str.strip, args.serotypes_list.split(',')))
    else:
        args.serotypes_list = None
    return args


def main(args):
    print(f"Starting visualization with method: {args.method}") 
    missing_label = args.params.get("missing_label", DEFAULT_MISSING_LABEL)
    label_column = args.params.get("label_column", DEFAULT_LABEL_COLUMN)
    downsample_size = args.params.get("downsample_size", DEFAULT_DOWNSAMPLE_SIZE)
    sep = args.params.get("sep", DEFAULT_SEP)
    
    print("Loading the data...")
    embeddings = np.load(args.embeddings)
    labels = pd.read_csv(args.labels, sep="\t")
    is_emb_npz = isinstance(embeddings, np.lib.npyio.NpzFile)
    if not is_emb_npz:
        assert embeddings.shape[0] == labels.shape[0], "Number of embeddings and labels do not match."

    labels = labels \
        .rename({label_column: "Serotype"}, axis=1) \
        .fillna(missing_label)
    indices_mask = np.ones(len(labels), dtype=bool) if args.show_untypable else labels["Serotype"] != missing_label
    if args.serotypes_list:
        serotypes_list = set(args.serotypes_list)
        serotypes_indices = labels["Serotype"].isin(serotypes_list)
        print(f"Filtering labels to include only: {serotypes_list}, {serotypes_indices.sum()} rows will be included.")
        indices_mask &= serotypes_indices
    
    if args.downsample:
        print("Downsampling the data for faster plotting...")
        downsample_indices = np.random.choice(np.where(indices_mask)[0], size=downsample_size, replace=False)
        indices_mask = np.zeros(len(labels), dtype=bool)
        indices_mask[downsample_indices] = True

    print("Calculating UMAP...")
    if is_emb_npz:
        cbl_prefix = lambda k: f"cbl{sep}{k}"
        X = np.array([embeddings[cbl_prefix(key)] for key in labels[indices_mask]["Public_name"]])
        # TODO Option to show non-cbl too
    else:
        X = embeddings[indices_mask]
    calc_fn = partial(
        calculate_umap if args.method.lower() == 'umap' else
        calculate_tsne if args.method.lower() == 'tsne' else
        calculate_pca,
        labels=labels[indices_mask],
        output_prefix=args.output_dir
    )
    _, embedding_df = calc_fn(X)
    
    print("Plotting UMAP...")
    plot_projection(embedding_df, args.method.upper(), args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
