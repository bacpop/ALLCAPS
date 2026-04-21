# !/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, ks_2samp

from ..logging_config import get_logger

logger = get_logger(__name__)

# Raises a RuntimeError if dvipng is not installed,
# but only after successfully writing the plot to file!
plt.rcParams.update(
    {
        "text.usetex": True,
        "svg.fonttype": "none",
        "ps.usedistiller": "xpdf",  # or 'none' or 'ghostscript'
    }
)

DEFAULT_FIGSIZE = (15, 15)


def compute_per_class_f1(cm):
    """
      - Precision for class i = cm[i, i] / sum of col i
      - Recall for class i    = cm[i, i] / sum of row i
      - F1 for class i = 2 * (prec * recall) / (prec + recall)
      - Balanced ACC = average recall across classes.
    Returns:
      f1_scores: a list of length N containing the F1 for each class i
      bal_acc:   the balanced accuracy
    """
    N = cm.shape[0]
    f1_scores, recalls = [], []

    for i in range(N):
        true_positives = cm[i, i]

        actual_count = cm[i, :].sum()
        recall = true_positives / actual_count if actual_count > 0 else 0.0
        recalls.append(recall)

        predicted_count = cm[:, i].sum()
        precision = true_positives / predicted_count if predicted_count > 0 else 0.0

        f1 = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        f1_scores.append(f1)

    bal_acc = np.mean(recalls)
    return f1_scores, bal_acc


def main(args):
    figsize = tuple(map(float, args.figsize.split(",")))
    plt.figure(figsize=figsize)
    kde_lines = []  # List to store KDE line objects
    f1_scores_distributions = []  # List to store F1 score distributions
    logger.info("Loading confusion matrices...")
    for cm_path in args.confusion_matrices:
        cm = pd.read_csv(cm_path, index_col=0).values
        f1_scores, bal_acc = compute_per_class_f1(cm)
        if len(f1_scores) == 1:
            continue
        f1_scores_distributions.append(f1_scores)

        xs = np.linspace(0, 1, 200)
        kde = gaussian_kde(f1_scores)
        n = len(f1_scores)
        ys = kde(xs) * n
        line_obj = plt.plot(xs, ys, label=f"{cm_path}")[0]
        kde_lines.append(line_obj)
        plt.axvline(bal_acc, linestyle="--", color=line_obj.get_color(), alpha=0.8)

    plt.xlabel("F1 Score"), plt.ylabel("Serotype Count")
    plt.title("F1 Score distributions per Method")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Messy stuff to get a clean legend
    dashed_line = Line2D(
        [0], [0], color="black", linestyle="--", label="Balanced Accuracy"
    )
    plt.legend(
        kde_lines + [dashed_line], args.legend + ["Balanced Accuracy"], loc="best"
    )

    plt.tight_layout()
    plt.savefig(args.output)
    plt.close()

    logger.info("Plot saved to %s.", args.output)

    if len(f1_scores_distributions) > 1:
        logger.info("Performing KS test...")
        f1_scores_baseline = f1_scores_distributions[0]
        f1_scores_method = f1_scores_distributions[1]
        ks_stat, p_value = ks_2samp(
            f1_scores_baseline, f1_scores_method, alternative="greater"
        )
        logger.info("\tKS test statistic: %s, p-value: %s", ks_stat, p_value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confusion_matrices",
        nargs="+",
        required=True,
        help="Paths to confusion matrix files (e.g., .csv). One plot line per file. The first file is the baseline.",
    )
    parser.add_argument(
        "--legend",
        nargs="+",
        default=None,
        help="List of legend labels for each confusion matrix.",
    )
    parser.add_argument(
        "--output",
        default="f1_distributions.pdf",
        help="Where to save the resulting plot.",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default=f"{DEFAULT_FIGSIZE[0]},{DEFAULT_FIGSIZE[1]}",
        help=f'Figure size as "width,height" (default: {DEFAULT_FIGSIZE})',
    )
    args = parser.parse_args()

    if args.legend is not None:
        assert len(args.confusion_matrices) == len(args.legend), (
            "Number of confusion matrices must match number of legend labels."
        )
    else:
        args.legend = [f"Data {i + 1}" for i in range(len(args.confusion_matrices))]
    return args


if __name__ == "__main__":
    main(parse_args())
