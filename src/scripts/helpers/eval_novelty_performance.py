import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from scipy.stats import percentileofscore


def report_roc(id_energies, query_vals):
    y_true = np.concatenate([np.zeros_like(id_energies), np.ones_like(query_vals)])
    scores = np.concatenate([id_energies, query_vals])

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_threshold = thresholds[best_idx]
    best_percentile = percentileofscore(id_energies, best_threshold)

    print(f"Best threshold (energy): {best_threshold:.4f}")
    print(f"Corresponding percentile of ID energies: {best_percentile:.2f}%")
    print(f"ROC AUC: {roc_auc:.4f}")

    print(f"At best threshold:")
    print(f"  True Positive Rate (Sensitivity): {tpr[best_idx]:.4f}")
    print(f"  False Positive Rate (1 - Specificity): {fpr[best_idx]:.4f}")

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], "k--")
    plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label='Best threshold')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: Novelty Energy Threshold")
    plt.legend()
    plt.grid(True)

    with open("energy_threshold_report.txt", "w") as f:
        f.write(f"Best threshold (energy): {best_threshold:.4f}\n")
        f.write(f"Corresponding percentile of ID energies: {best_percentile:.2f}%\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"At best threshold:\n")
        f.write(f"  True Positive Rate (Sensitivity): {tpr[best_idx]:.4f}\n")
        f.write(f"  False Positive Rate (1 - Specificity): {fpr[best_idx]:.4f}\n")
    plt.savefig("roc_curve_novelty_energy_threshold.pdf", dpi=300)

    plt.show()


def plot_energies(id_energies, query_vals):
    plt.figure(figsize=(15,6))

    plt.hist(id_energies, bins=100, alpha=0.7, label='CBL')
    plt.hist(query_vals, bins=100, alpha=0.9, label='Query')
    # plt.hist(df['energy_serotype'][~df["is_cbl"]], bins=100, alpha=0.4, label='NON-CBL')
    # plt.hist(df['energy_serotype'], bins=100, alpha=0.7)

    plt.xlabel('Energy')
    plt.ylabel('Frequency')
    plt.xticks([x/10.0 for x in range(-140, -40, 2)])
    plt.xticks(rotation=45)
    plt.title('Distribution of Serotype Energies')
    plt.legend()

    plt.savefig("energy_histogram.pdf", dpi=300)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_path", required=True, help="Path to the CSV file containing energies and labels")
    parser.add_argument("--query_path", required=True, help="Path to the CSV file containing query energies")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(args.df_path, index_col=0)
    query_energies = pd.read_csv(args.query_path, index_col=0)
    df["is_cbl"] = ~df["Serotype"].str.contains("NON-CBL")

    id_energies = df.loc[df["is_cbl"], "energy_serotype"].to_numpy()
    query_vals = query_energies["novelty_confidence"].to_numpy()
    
    plot_energies(id_energies, query_vals)
    report_roc(id_energies, query_vals)
