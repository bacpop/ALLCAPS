"""k=1 nearest-neighbour diagnostic across LOO folds — "when we can't flag a
novel serotype, does it at least land next to the right neighbourhood?".

For every leave-one-serotype-out fold, the held-out serotype's query contigs
were scored against the ID index and tagged (in ``knn_query_distances.csv``, by
``scripts.knn_ood predict``) with their single closest ID neighbour's serotype
and genogroup (``nn_serotype`` / ``nn_genogroup``). This script aggregates those
tags to answer:

  1. Per held-out serotype, what fraction of its query contigs land next to a
     serotype in the SAME genogroup ("genogroup hit rate")? A high hit rate is
     the structured-embedding story even where novelty detection fails.
  2. Which genogroup do the novels actually land in — a genogroup-level
     confusion matrix (held-out genogroup → nearest-neighbour genogroup),
     row-normalised so the diagonal reads as the hit rate.

This reads only the per-fold ``knn_query_distances.csv`` files (cheap, local);
it does NOT re-embed or re-fit anything.

Usage:
  python -m scripts.helpers.analyze_knn_nearest_neighbor \\
      --loo_dir /path/to/gps-loo \\
      --output_dir results/knn-nn
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless plotting on the cluster

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .knn_k_sweep import _genogroup_colors, _restore_serotype, _save_both
from ..logging_config import get_logger
from ..utils import map_serotype_to_group

logger = get_logger(__name__)

REQUIRED_COLS = ("nn_serotype", "nn_genogroup")
SMALL_N_QUERY = 5  # folds with fewer query contigs are flagged unreliable


# ──────────────────────────── aggregation ────────────────────────────


def _collect(loo_dir: Path, fold_glob: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk fold subdirs, read each ``knn_query_distances.csv``, and return
    ``(per_query, per_fold)``.

    ``per_query`` is every query contig with its held-out serotype/genogroup and
    the nearest-neighbour tags. ``per_fold`` is the one-row-per-serotype summary.
    """
    per_query_frames: list[pd.DataFrame] = []
    fold_rows: list[dict] = []

    fold_dirs = sorted(d for d in loo_dir.glob(fold_glob) if d.is_dir())
    logger.info("Found %d candidate fold directories under %s", len(fold_dirs), loo_dir)

    for fold_dir in fold_dirs:
        csv_path = fold_dir / "knn_query_distances.csv"
        if not csv_path.exists():
            logger.warning("Fold %s: missing %s — skipping", fold_dir.name, csv_path.name)
            continue
        df = pd.read_csv(csv_path)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            logger.warning(
                "Fold %s: %s lacks %s (predicted before the k=1 NN report was added?) — skipping",
                fold_dir.name, csv_path.name, missing,
            )
            continue
        if len(df) == 0:
            logger.warning("Fold %s: no query rows — skipping", fold_dir.name)
            continue

        held_out = _restore_serotype(fold_dir.name)
        held_out_geno = map_serotype_to_group(held_out)
        df = df.copy()
        df["held_out_serotype"] = held_out
        df["held_out_genogroup"] = held_out_geno
        df["genogroup_match"] = df["nn_genogroup"].astype(str) == held_out_geno
        per_query_frames.append(df)

        nn_sero_counts = df["nn_serotype"].astype(str).value_counts()
        nn_geno_counts = df["nn_genogroup"].astype(str).value_counts()
        n = len(df)
        fold_rows.append({
            "held_out_serotype": held_out,
            "held_out_genogroup": held_out_geno,
            "n_query": int(n),
            "genogroup_hit_rate": float(df["genogroup_match"].mean()),
            "modal_nn_serotype": nn_sero_counts.index[0],
            "modal_nn_serotype_frac": float(nn_sero_counts.iloc[0] / n),
            "modal_nn_genogroup": nn_geno_counts.index[0],
            "modal_nn_genogroup_frac": float(nn_geno_counts.iloc[0] / n),
            "median_nn_distance": float(df["nn_distance"].median())
            if "nn_distance" in df.columns else float("nan"),
        })

    if not fold_rows:
        raise SystemExit(
            "No usable folds found. Ensure knn_query_distances.csv files exist and "
            "carry the nn_serotype/nn_genogroup columns (re-run the KNN predict stage)."
        )

    per_query = pd.concat(per_query_frames, ignore_index=True)
    per_fold = pd.DataFrame(fold_rows).sort_values("genogroup_hit_rate", ascending=False)
    return per_query, per_fold


# ──────────────────────────── plots ────────────────────────────


def _plot_hit_rate(per_fold: pd.DataFrame, out_stem: Path) -> None:
    """Sorted per-serotype bar of the genogroup hit rate, coloured by the held-out
    genogroup; ``*`` marks folds with few query contigs."""
    sub = per_fold.sort_values("genogroup_hit_rate", ascending=False).reset_index(drop=True)
    colors_map = _genogroup_colors(list(sub["held_out_genogroup"]))
    colors = [colors_map[g] for g in sub["held_out_genogroup"]]
    labels = [
        f"{s} *" if n < SMALL_N_QUERY else str(s)
        for s, n in zip(sub["held_out_serotype"], sub["n_query"])
    ]

    fig, ax = plt.subplots(figsize=(max(8, 0.20 * len(sub)), 5), dpi=150)
    ax.bar(range(len(sub)), sub["genogroup_hit_rate"], color=colors)
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Fraction of query contigs with NN in same genogroup")
    ax.set_title("k=1 nearest-neighbour genogroup hit rate per held-out serotype")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.text(
        0.5, -0.09,
        "Nearest ID neighbour of each held-out (novel) contig. High bar = the "
        "embedding places the novel next to its own genogroup even when it can't "
        f"be flagged as novel. * n_query<{SMALL_N_QUERY} (unreliable).",
        ha="center", fontsize=8, style="italic", color="#444",
    )
    _save_both(fig, out_stem)


def _genogroup_confusion(per_query: pd.DataFrame) -> pd.DataFrame:
    """Row-normalised held-out-genogroup → NN-genogroup matrix. Columns are
    ordered to put the row genogroups first (diagonal), then any extra NN-only
    genogroups."""
    ct = pd.crosstab(
        per_query["held_out_genogroup"],
        per_query["nn_genogroup"].astype(str),
        normalize="index",
    )
    row_order = sorted(ct.index)
    col_order = row_order + sorted(set(ct.columns) - set(row_order))
    return ct.reindex(index=row_order, columns=col_order, fill_value=0.0)


def _plot_confusion(conf: pd.DataFrame, out_stem: Path) -> None:
    n_rows, n_cols = conf.shape
    fig, ax = plt.subplots(
        figsize=(max(7, 0.42 * n_cols), max(6, 0.42 * n_rows)), dpi=150
    )
    im = ax.imshow(conf.to_numpy(), aspect="auto", cmap="magma_r", vmin=0, vmax=1)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(conf.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(conf.index, fontsize=6)
    ax.set_xlabel("Nearest-neighbour genogroup")
    ax.set_ylabel("Held-out (novel) genogroup")
    ax.set_title("Genogroup confusion of k=1 nearest neighbour (row-normalised)")
    # Annotate only non-trivial cells to avoid clutter.
    vals = conf.to_numpy()
    for i in range(n_rows):
        for j in range(n_cols):
            v = vals[i, j]
            if v >= 0.10:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if v > 0.5 else "#222222")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Fraction of contigs")
    _save_both(fig, out_stem)


# ──────────────────────────── CLI ────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--loo_dir", required=True,
                   help="Root directory; one subdir per held-out serotype")
    p.add_argument("--output_dir", required=True, help="Where to write CSVs + plots")
    p.add_argument("--fold_glob", default="*",
                   help="Glob for fold subdir names (e.g. '6*' to subset)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    loo_dir = Path(args.loo_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_query, per_fold = _collect(loo_dir, args.fold_glob)
    conf = _genogroup_confusion(per_query)

    per_fold.to_csv(output_dir / "knn_nn_summary.csv", index=False)
    conf.to_csv(output_dir / "knn_nn_genogroup_confusion.csv")

    _plot_hit_rate(per_fold, output_dir / "knn_nn_genogroup_hit_rate")
    _plot_confusion(conf, output_dir / "knn_nn_genogroup_confusion")

    median_hit = float(per_fold["genogroup_hit_rate"].median())
    frac_high = float((per_fold["genogroup_hit_rate"] >= 0.5).mean())
    logger.info("Wrote outputs to %s", output_dir)
    logger.info("  knn_nn_summary.csv (%d folds)", len(per_fold))
    logger.info("  knn_nn_genogroup_confusion.csv (%d x %d)", *conf.shape)
    logger.info("  knn_nn_genogroup_hit_rate.{pdf,png}")
    logger.info("  knn_nn_genogroup_confusion.{pdf,png}")
    logger.info("Headline: median genogroup hit rate = %.2f; %.0f%% of folds land "
                "in their own genogroup >50%% of the time.",
                median_hit, 100 * frac_high)


if __name__ == "__main__":
    main()
