import argparse

import umap
import umap.plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MISSING_LABEL = "Non-typeable"  # TODO: Make this a parameter
DOWNSAMPLE_SIZE = 1000  # TODO: Make this a parameter
LABEL_COLUMN = "serotype"  # TODO: Do sth about it

def calculate_umap(embeddings, labels, output_prefix):
    out_path = output_prefix + ".umap.csv"
    reducer = umap.UMAP(random_state=42)  # Use no seed for parallelism
    mapper = reducer.fit(embeddings)
    UMAP_embedding = reducer.transform(embeddings)
    UMAP_embedding_df = pd.DataFrame(UMAP_embedding)

    UMAP_embedding_df.insert(0, 'Sample', labels.index)
    UMAP_embedding_df.insert(1, 'Serotype', labels.Serotype)
    UMAP_embedding_df.columns = ['Sample', 'Serotype', 'UMAP1', 'UMAP2']
    # UMAP_embedding_df.to_csv(out_path + ".umap.csv", index=False)
    return mapper, UMAP_embedding_df


def plot_umap(mapper, df, output_path):
    """
    Plot UMAP visualization of the dataframe.

    Parameters:
    - data: DataFrame containing the data to visualize. The columns are ['UMAP1', 'UMAP2', 'Serotype'].
    - output_path: Path to save the UMAP plot.
    """
    # Plotting
    # plt.figure(figsize=(10, 8))
    # sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='Serotype', palette='viridis', alpha=0.7)
    p = umap.plot.points(mapper, labels=df['Serotype'], theme='fire')  # TODO: rearrange the colors
    plt.title('UMAP Visualization')
    plt.savefig(output_path)  # , dpi=300, bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="UMAP visualization of data.")
    parser.add_argument('--embeddings', type=str, required=True, help='Path to the embeddings.')
    parser.add_argument('--labels', type=str, required=True, help='Path to the labels file (CSV format).')
    parser.add_argument('--output', type=str, required=True, help='Path to save the UMAP plot.')
    parser.add_argument('--title', type=str, default="UMAP", help='Title of the plot.')
    parser.add_argument('--show-untypable', type=bool, default=False, help='Include data with missing label in the plot.')
    parser.add_argument('--downsample', type=bool, default=False, help='Downsample the data for faster plotting.')
    return parser.parse_args()


def main(args):
    # Load the data
    embeddings = np.load(args.embeddings)
    labels = pd.read_csv(args.labels, sep="\t", index_col=0)
    print(labels.head())
    assert embeddings.shape[0] == labels.shape[0], "Number of embeddings and labels do not match."

    labels['Serotype'] = labels[LABEL_COLUMN].fillna(MISSING_LABEL)
    indice_mask = np.ones(len(labels), dtype=bool) if args.show_untypable else labels['Serotype'] != MISSING_LABEL
    if args.downsample:
        print("Downsample the data for faster plotting...")
        downsample_indices = np.random.choice(np.where(indice_mask)[0], size=DOWNSAMPLE_SIZE, replace=False)
        indice_mask = np.zeros(len(labels), dtype=bool)
        indice_mask[downsample_indices] = True
    
    print("Calculating UMAP...")
    mapper, UMAP_embedding_df = calculate_umap(embeddings[indice_mask], labels[indice_mask], args.output)
    print("Plotting UMAP...")
    plot_umap(mapper, UMAP_embedding_df, args.output)    


if __name__ == "__main__":
    args = parse_args()
    main(args)
