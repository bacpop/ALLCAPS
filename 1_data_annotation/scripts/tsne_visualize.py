#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="t-SNE")
    args = parser.parse_args()

    X = np.load(args.embeddings)  # shape (N, D)
    y = pd.read_csv(args.labels, sep="\t")["serotype"]  # shape (N,)

    y = y.fillna("Missing").replace("Non-typeable", "Missing")
    y = y.values.tolist()
    
    assert len(X) == len(y), "Mismatch between embeddings and labels length"

    tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="random", random_state=42)
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(6, 6))
    unique_labels = list(set(y))
    color_map = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    label_to_color = dict(zip(unique_labels, color_map))
    label_to_color["Missing"] = "gray"  # Assign gray to "Missing"
    label_to_color["Missing"] = "gray"

    for i, point in enumerate(X_2d):
        lbl = y[i]
        plt.scatter(point[0], point[1], color=label_to_color[lbl], alpha=0.7)
    
    for lbl in unique_labels:  # Create legend (one entry per label)
        plt.scatter([], [], color=label_to_color[lbl], label=lbl)
    plt.legend(loc="best", title="Label")

    plt.title(args.title)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.savefig(args.output)
    plt.close()
    print(f"Saved t-SNE plot to {args.output}")

if __name__ == "__main__":
    main()
