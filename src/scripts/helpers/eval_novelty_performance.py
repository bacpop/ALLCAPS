import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from scipy.stats import percentileofscore

plt.rcParams.update({
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.edgecolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "font.size": 10,
})


def report_roc(id_energies, query_vals, output_dir):
    y_true = np.concatenate([np.zeros_like(id_energies), np.ones_like(query_vals)])
    scores = np.concatenate([id_energies, query_vals])

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_threshold = thresholds[best_idx]
    best_percentile = percentileofscore(id_energies, best_threshold)

    summary_lines = [
        f"Best threshold (energy): {best_threshold:.4f}",
        f"Corresponding percentile of ID energies: {best_percentile:.2f}%",
        f"ROC AUC: {roc_auc:.4f}",
        "At best threshold:",
        f"  True Positive Rate (Sensitivity): {tpr[best_idx]:.4f}",
        f"  False Positive Rate (1 - Specificity): {fpr[best_idx]:.4f}",
    ]
    print("\n".join(summary_lines))

    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=150)
    ax.plot(fpr, tpr, color="#0b2545", linewidth=2.2, label=f"Model (AUC = {roc_auc:.2f})")
    ax.fill_between(fpr, tpr, fpr, color="#0b2545", alpha=0.08)
    ax.plot([0, 1], [0, 1], color="#9aa0a6", linestyle="--", linewidth=1.0, label="Random chance")

    best_point = (fpr[best_idx], tpr[best_idx])
    ax.scatter(*best_point, color="#c44536", s=36, zorder=5, label="Youden max")

    def _offset(val, delta):
        return float(np.clip(val + delta, 0.05, 0.95))

    annotation_text = (
        f"Threshold: {best_threshold:.2f}\n"
        f"TPR: {tpr[best_idx]:.1%}\n"
        f"FPR: {fpr[best_idx]:.1%}"
    )
    ax.annotate(
        annotation_text,
        xy=best_point,
        xytext=(_offset(best_point[0], 0.12), _offset(best_point[1], -0.18)),
        arrowprops=dict(arrowstyle="-", color="#444444", linewidth=0.8),
        ha="left",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.85)
    )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", "box")
    ax.tick_params(axis="both", length=4, width=0.8, colors="#222222")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC Curve — Serotype {serotype}", pad=18)
    ax.legend(frameon=False, loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir + f"/roc_curve_novelty_serotype_{serotype}.pdf", bbox_inches="tight", transparent=False)

    with open(output_dir + "/energy_threshold_report.txt", "w") as f:
        f.write("\n".join(summary_lines))


def plot_energies(id_energies, query_vals, output_dir):
    y_true = np.concatenate([np.zeros_like(id_energies), np.ones_like(query_vals)])
    scores = np.concatenate([id_energies, query_vals])
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    best_threshold = thresholds[np.argmax(tpr - fpr)]

    fig, ax = plt.subplots(figsize=(15, 6))

    ax.hist(id_energies, bins=100, alpha=0.7, label='CBL')
    ax.hist(query_vals, bins=100, alpha=0.9, label='Query', color='#c44536')  # A dishonest trick: np.tile(query_vals, 2) to amplify for visualization
    # plt.hist(df['energy_serotype'][~df["is_cbl"]], bins=100, alpha=0.4, label='NON-CBL')

    ax.axvline(best_threshold, color='black', linestyle='--', linewidth=1.2, label=f'Threshold ({best_threshold:.2f})')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('Energy')
    ax.set_ylabel('Frequency')
    ax.set_xticks([x/10.0 for x in range(-130, -50, 2)])
    plt.xticks(rotation=45)
    ax.set_title('Distribution of Serotype Energies')
    ax.legend()

    fig.savefig(output_dir + f"/energy_histogram_{serotype}.pdf", dpi=300)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_path", required=True, help="Path to the CSV file containing energies and labels")
    parser.add_argument("--query_path", required=True, help="Path to the CSV file containing query energies")
    parser.add_argument("--output_dir", required=True, help="Directory to save output files")
    parser.add_argument("--serotype", required=True, help="Serotype")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    serotype = args.serotype

    df = pd.read_csv(args.df_path, index_col=0)
    query_energies = pd.read_csv(args.query_path, index_col=0)
    df["is_cbl"] = ~df["Serotype"].str.contains("NON-CBL")

    id_energies = df.loc[df["is_cbl"], "energy_serotype"].to_numpy()
    query_vals = query_energies["novelty_confidence"].to_numpy()
    
    plot_energies(id_energies, query_vals, args.output_dir)
    report_roc(id_energies, query_vals, args.output_dir)
