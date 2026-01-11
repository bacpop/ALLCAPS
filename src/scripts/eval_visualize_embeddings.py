import json
import argparse
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from utils import load_data, map_serotype_to_group
from consts import DEFAULT_MISSING_LABEL, DEFAULT_LABEL_COLUMN

DEFAULT_FIGSIZE = (15, 15)

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


def plot_projection(df, method, output_prefix, params={}):
    """
    Plot the visualization of the dataframe.

    Parameters:
    - data: DataFrame containing the data to visualize. The columns are ['{method}1', '{method}2', 'Serotype'].
    - output_path: Path to save the UMAP plot.
    """

    serotypes = df['Serotype']  # .apply(map_serotype_to_group)
    unique_serotypes = serotypes.unique()
    
    # Setup plot params
    figsize = params["figsize"]
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_serotypes)))  # TODO explore colormaps
    color_map = dict(zip(unique_serotypes, colors))

    plt.figure(figsize=figsize)
    for serotype, color in color_map.items():
        subset = df[serotypes == serotype]
        plt.scatter(subset[f'{method}1'], subset[f'{method}2'], label=serotype, color=color, alpha=0.7, s=10)

    plt.title(f'{method} Visualization')
    plt.xlabel(f'{method}1'), plt.ylabel(f'{method}2')

    plt.legend(title="Serogroups", loc='best', ncol=2, fontsize='small', markerscale=2.0, facecolor='darkgray')
    plt.gcf().patch.set_facecolor('black')
    plt.gca().set_facecolor('black')

    plt.tight_layout()
    plt.savefig(output_prefix + f'_{method.lower()}.pdf')
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="2D visualization of data.")
    parser.add_argument('--embeddings', type=str, required=True, help='Path to the embeddings.')
    parser.add_argument('--labels', type=str, required=True, help='Path to the labels file (CSV format).')
    parser.add_argument('--output_prefix', type=str, required=True, help='Path to save the figure (prefix).')
    parser.add_argument('--method', type=str, choices=['umap', 'tsne', 'pca'], default='umap',
                        help='Dimensionality reduction method to use for visualization.')
    parser.add_argument('--params', type=str, default="{}", help='JSON string of parameters.')
    parser.add_argument('--show_noncbl', action='store_true', default=False, help='Include non capsule sequences in the plot.')
    parser.add_argument('--downsample', type=int, default=None, help='Downsample the data for faster plotting.')
    parser.add_argument('--serotypes_list', type=str, default=None, help='Comma-separated list of serotypes to include in the plot.')
    parser.add_argument('--figsize', type=str, default=f"{DEFAULT_FIGSIZE[0]},{DEFAULT_FIGSIZE[1]}", help=f'Figure size as "width,height" (default: {DEFAULT_FIGSIZE})')
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

    try:
        figsize = tuple(map(float, args.figsize.split(',')))
    except ValueError:
        print(f"Error: figsize must be in 'width,height' format. Using default {DEFAULT_FIGSIZE}.")
        figsize = DEFAULT_FIGSIZE
    
    args.params["figsize"] = figsize
    if args.serotypes_list:
        args.serotypes_list = list(map(str.strip, args.serotypes_list.split(',')))
    else:
        args.serotypes_list = None
    return args


def main(args):
    print(f"Starting visualization with method: {args.method}") 
    missing_label, label_column = DEFAULT_MISSING_LABEL, DEFAULT_LABEL_COLUMN
    
    print("Loading the data...")
    embeddings, labels = load_data(args.embeddings, args.labels, missing_label=missing_label)
    labels = labels \
        .rename({label_column: "Serotype"}, axis=1) \
        .fillna(missing_label)

    indices_mask = labels["Serotype"] != missing_label
    if args.serotypes_list:
        serotypes_list = set(args.serotypes_list)
        serotypes_indices = labels["Serotype"].isin(serotypes_list)
        print(f"Filtering labels to include only: {serotypes_list}, {serotypes_indices.sum()} rows will be included.")
        indices_mask &= serotypes_indices
    
    if args.downsample:
        print("Downsampling the data for faster plotting...")
        downsample_indices = np.random.choice(np.where(indices_mask)[0], size=args.downsample, replace=False)
        indices_mask = np.zeros(len(labels), dtype=bool)
        indices_mask[downsample_indices] = True

    print("Calculating...")
    if not args.show_noncbl:
        print("Using only capsule locus embeddings (cbl).")
        indices_mask &= labels["Is_capsule"]

    calc_fn = partial(
        calculate_umap if args.method.lower() == 'umap' else
        calculate_tsne if args.method.lower() == 'tsne' else
        calculate_pca,
        labels=labels[indices_mask],
        output_prefix=args.output_prefix
    )
    _, embedding_df = calc_fn(embeddings[indices_mask])

    print("Plotting...")
    plot_projection(embedding_df, args.method.upper(), args.output_prefix, args.params)


if __name__ == "__main__":
    args = parse_args()
    main(args)
