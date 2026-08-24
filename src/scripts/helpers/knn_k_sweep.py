"""KNN-OOD k-sweep diagnostic — answers "is our AUROC ceiling set by k=50 or by features?".

For every LOO fold, sweeps a fixed k grid and computes per-(fold, k) AUROC.
The KNN index is fit once per fold (with n_neighbors = max(k)+1) and the
distance matrix is sliced for every k — sweep cost ≈ single max(k) query.

The output answers three diagnostic questions:
  1. How sensitive is KNN-AUROC to k?
  2. What's the per-fold AUROC ceiling under k tuning vs the current k=50?
  3. Per serotype, what balanced-accuracy ceiling is achievable at the optimal
     (k, threshold), and what does the deployable operating point actually get?
     Two plots — threshold_accuracy_per_serotype (ceiling: per-serotype optimal
     k & threshold) and threshold_accuracy_per_serotype_deploy (deployment: fixed
     k, p99-of-ID threshold set without the novels). Balanced accuracy is used
     because ID vastly outnumbers novel, making raw accuracy degenerate.

Per-fold "best k" is a CEILING, not a deployment hyperparameter. At inference
time we don't know which serotype is held out, so picking k per-fold is
hyperparameter-on-test-set leakage. For deployment, use nested CV instead.

Usage:
  micromamba run -n ebi_env python -m scripts.helpers.knn_k_sweep \\
      --loo_dir /path/to/split_loo_results \\
      --output_dir results/knn_k_sweep \\
      --k_grid 1,5,10,20,50,100,200,500,1000
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors

from ..knn_ood import _load_id_embeddings, _load_query_embeddings
from ..logging_config import get_logger
from ..utils import map_serotype_to_group

logger = get_logger(__name__)

DEFAULT_K_GRID = (1, 5, 10, 20, 50, 100, 200, 500, 1000)
DEFAULT_REFERENCE_K = 50
DEFAULT_DEPLOY_K = 5
DEFAULT_DEPLOY_PERCENTILE = 99.0

plt.rcParams.update({
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.edgecolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "font.size": 10,
})


def _restore_serotype(safe_name: str) -> str:
    return safe_name.replace("-", "/")


def _kth_nn_distances(
    nn: NearestNeighbors,
    X: np.ndarray,
    k_grid: list[int],
    query_is_id: bool,
) -> dict[int, np.ndarray]:
    """One brute-force query at max(k_grid) (+1 if querying ID, to drop self-match);
    return dict mapping k → distance to k-th NN. Slicing is free."""
    max_k = max(k_grid)
    n_search = max_k + 1 if query_is_id else max_k
    dists, _ = nn.kneighbors(X, n_neighbors=n_search, return_distance=True)
    # If querying ID, drop column 0 (self-match, distance 0)
    if query_is_id:
        dists = dists[:, 1:]
    return {k: dists[:, k - 1] for k in k_grid}


def _best_balanced_accuracy(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Best balanced accuracy = max over thresholds of (TPR + TNR) / 2, and the
    distance threshold that achieves it. Positive class (label 1) = novel, so
    TPR is novel-recall and (1-FPR) is ID-specificity. Threshold-optimal on the
    held-out novels ⇒ this is a CEILING (leaks the operating point), consistent
    with the best-k ceiling reported alongside it."""
    fpr, tpr, thr = roc_curve(y_true, scores)
    bal_acc = (tpr + (1.0 - fpr)) / 2.0
    i = int(np.argmax(bal_acc))
    return float(bal_acc[i]), float(thr[i])


def _deployment_operating_point(
    id_scores: np.ndarray, novel_scores: np.ndarray, percentile: float
) -> tuple[float, float, float, float]:
    """Deployment operating point at a FIXED k: threshold = ``percentile`` of the
    ID (train LOO) k-th-NN distances — set WITHOUT looking at the novels, so it's
    honest/deployable (the convention in knn_ood.py). Returns
    (bal_acc, tpr, tnr, threshold). Note TNR ≈ percentile/100 by construction, so
    the balanced accuracy tracks the detection rate (TPR)."""
    threshold = float(np.percentile(id_scores, percentile))
    tpr = float(np.mean(novel_scores > threshold))   # novels flagged novel
    tnr = float(np.mean(id_scores <= threshold))      # ID kept ID (≈ percentile/100)
    bal_acc = (tpr + tnr) / 2.0
    return bal_acc, tpr, tnr, threshold


def _sweep_one_fold(
    fold_dir: Path,
    k_grid: list[int],
    distance_metric: str,
    deploy_k: int,
    deploy_percentile: float,
) -> tuple[pd.DataFrame, dict] | None:
    """Run the k-sweep on one LOO subdir. Returns (per-k DataFrame, deployment
    record) or None if the fold can't be loaded. The deployment record captures
    the fixed-k / percentile-threshold operating point for the companion plot."""
    fold_safe = fold_dir.name
    serotype = _restore_serotype(fold_safe)
    genogroup = map_serotype_to_group(serotype)

    inference_path = fold_dir / "inference_results.npz"
    labels_path = fold_dir / "final_metadata.csv"
    query_path = fold_dir / "query_embeddings.npz"
    for required in (inference_path, labels_path, query_path):
        if not required.exists():
            logger.warning("Fold %s: missing %s — skipping", fold_safe, required.name)
            return None

    X_id, _, _ = _load_id_embeddings(str(inference_path), str(labels_path))
    X_novel, _ = _load_query_embeddings(str(query_path))

    # Clamp k grid to training-set size (drop k values bigger than n_id-1)
    effective_grid = [k for k in k_grid if k <= len(X_id) - 1]
    if len(effective_grid) < len(k_grid):
        skipped = [k for k in k_grid if k not in effective_grid]
        logger.warning("Fold %s: n_id=%d, skipping k=%s (too large)",
                       fold_safe, len(X_id), skipped)
    if not effective_grid:
        logger.warning("Fold %s: no valid k values — skipping", fold_safe)
        return None

    max_k = max(effective_grid)
    nn = NearestNeighbors(
        n_neighbors=max_k + 1,
        metric=distance_metric,
        algorithm="brute",
    )
    nn.fit(X_id)
    id_dists = _kth_nn_distances(nn, X_id, effective_grid, query_is_id=True)
    novel_dists = _kth_nn_distances(nn, X_novel, effective_grid, query_is_id=False)

    rows = []
    y_true = np.concatenate([np.zeros(len(X_id)), np.ones(len(X_novel))])
    for k in effective_grid:
        scores = np.concatenate([id_dists[k], novel_dists[k]])
        auroc = float(roc_auc_score(y_true, scores))
        bal_acc, bal_acc_threshold = _best_balanced_accuracy(y_true, scores)
        rows.append({
            "fold": serotype,
            "fold_safe_name": fold_safe,
            "genogroup": genogroup,
            "n_id": int(len(X_id)),
            "n_novel": int(len(X_novel)),
            "k": int(k),
            "auroc": auroc,
            "bal_acc": bal_acc,
            "bal_acc_threshold": bal_acc_threshold,
        })
    # Deployment operating point at the fixed deploy_k (honest threshold).
    if deploy_k in id_dists:
        d_bal, d_tpr, d_tnr, d_thr = _deployment_operating_point(
            id_dists[deploy_k], novel_dists[deploy_k], deploy_percentile)
    else:
        d_bal = d_tpr = d_tnr = d_thr = float("nan")
        logger.warning("Fold %s: deploy_k=%d not in effective grid — deployment point skipped",
                       fold_safe, deploy_k)
    deploy_record = {
        "fold": serotype,
        "deploy_k": int(deploy_k),
        "deploy_threshold_percentile": float(deploy_percentile),
        "deploy_threshold": d_thr,
        "deploy_tpr": d_tpr,
        "deploy_tnr": d_tnr,
        "deploy_bal_acc": d_bal,
    }

    logger.info("Fold %s: k-sweep done (n_id=%d, n_novel=%d, k range %d-%d)",
                fold_safe, len(X_id), len(X_novel),
                min(effective_grid), max(effective_grid))
    return pd.DataFrame(rows), deploy_record


# ──────────────────────────── aggregates ────────────────────────────


def _summary_per_k(per_fold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, sub in per_fold.groupby("k"):
        vals = sub["auroc"].dropna().to_numpy()
        if len(vals) == 0:
            continue
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        rows.append({
            "k": int(k),
            "n_folds": len(vals),
            "auroc_mean": float(vals.mean()),
            "auroc_median": float(q50),
            "auroc_p25": float(q25),
            "auroc_p75": float(q75),
            "auroc_min": float(vals.min()),
            "auroc_max": float(vals.max()),
            "frac_folds_above_0.9": float((vals > 0.9).mean()),
        })
    return pd.DataFrame(rows).sort_values("k")


def _best_per_fold(per_fold: pd.DataFrame, reference_k: int) -> pd.DataFrame:
    rows = []
    for fold, sub in per_fold.groupby("fold"):
        best_idx = sub["auroc"].idxmax()
        best_row = sub.loc[best_idx]
        ref = sub[sub["k"] == reference_k]
        auroc_at_ref = float(ref.iloc[0]["auroc"]) if len(ref) else float("nan")
        # Balanced-accuracy-optimal operating point (jointly over k and threshold).
        # "optimal k" for the threshold-accuracy plot is the k that maximizes the
        # achievable balanced accuracy — which may differ from the AUROC-optimal k.
        ba_idx = sub["bal_acc"].idxmax()
        ba_row = sub.loc[ba_idx]
        rows.append({
            "fold": fold,
            "genogroup": best_row["genogroup"],
            "n_novel": int(best_row["n_novel"]),
            "best_k": int(best_row["k"]),
            "best_auroc": float(best_row["auroc"]),
            "auroc_at_reference_k": auroc_at_ref,
            "reference_k": int(reference_k),
            "delta_vs_reference": float(best_row["auroc"] - auroc_at_ref) if np.isfinite(auroc_at_ref) else float("nan"),
            "best_bal_acc": float(ba_row["bal_acc"]),
            "best_bal_acc_k": int(ba_row["k"]),
            "best_bal_acc_threshold": float(ba_row["bal_acc_threshold"]),
        })
    return pd.DataFrame(rows).sort_values("delta_vs_reference", ascending=False)


# ──────────────────────────── plots ────────────────────────────


_DIAGNOSTIC_FOOTNOTE = (
    "Per-fold best k is a diagnostic ceiling, NOT a deployment hyperparameter. "
    "For deployment, use nested CV."
)


def _genogroup_colors(genogroups: list[str]) -> dict[str, tuple]:
    cmap = plt.get_cmap("tab20")
    uniq = sorted(set(genogroups))
    return {g: cmap(i % cmap.N) for i, g in enumerate(uniq)}


def _save_both(fig, path_stem: Path) -> None:
    fig.savefig(str(path_stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(path_stem) + ".png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def _plot_per_fold_spaghetti(per_fold: pd.DataFrame, out_stem: Path) -> None:
    folds = per_fold["fold"].unique()
    geno_by_fold = (per_fold.drop_duplicates("fold").set_index("fold")["genogroup"]).to_dict()
    colors = _genogroup_colors(list(geno_by_fold.values()))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for fold in folds:
        sub = per_fold[per_fold["fold"] == fold].sort_values("k")
        ax.plot(sub["k"], sub["auroc"], color=colors[geno_by_fold[fold]],
                alpha=0.35, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("k (log scale)")
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC vs k — one line per LOO fold (colored by genogroup)")
    ax.set_ylim(0, 1.02)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.text(0.5, -0.02, _DIAGNOSTIC_FOOTNOTE, ha="center", fontsize=8, style="italic", color="#444")
    _save_both(fig, out_stem)


def _plot_aggregated(summary: pd.DataFrame, per_fold: pd.DataFrame, out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    # IQR + p5-p95 bands, computed per-k from per_fold (not summary, which only has IQR)
    ks = sorted(per_fold["k"].unique())
    p5 = [np.percentile(per_fold[per_fold["k"] == k]["auroc"], 5) for k in ks]
    p25 = [np.percentile(per_fold[per_fold["k"] == k]["auroc"], 25) for k in ks]
    p50 = [np.percentile(per_fold[per_fold["k"] == k]["auroc"], 50) for k in ks]
    p75 = [np.percentile(per_fold[per_fold["k"] == k]["auroc"], 75) for k in ks]
    p95 = [np.percentile(per_fold[per_fold["k"] == k]["auroc"], 95) for k in ks]
    ax.fill_between(ks, p5, p95, color="#0b2545", alpha=0.10, label="p5–p95")
    ax.fill_between(ks, p25, p75, color="#0b2545", alpha=0.25, label="IQR")
    ax.plot(ks, p50, color="#0b2545", linewidth=2.2, label="median")
    ax.set_xscale("log")
    ax.set_xlabel("k (log scale)")
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC vs k — aggregated across folds")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.text(0.5, -0.02, _DIAGNOSTIC_FOOTNOTE, ha="center", fontsize=8, style="italic", color="#444")
    _save_both(fig, out_stem)


def _plot_best_k_distribution(best_per_fold: pd.DataFrame, k_grid: list[int], out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    counts = best_per_fold["best_k"].value_counts().reindex(k_grid, fill_value=0)
    positions = range(len(k_grid))
    ax.bar(positions, counts.values, color="#0b2545")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(k) for k in k_grid])
    ax.set_xlabel("k")
    ax.set_ylabel("Number of folds")
    ax.set_title("Per-fold best-k distribution")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.text(0.5, -0.04, _DIAGNOSTIC_FOOTNOTE, ha="center", fontsize=8, style="italic", color="#444")
    _save_both(fig, out_stem)


def _plot_tuning_headroom(best_per_fold: pd.DataFrame, out_stem: Path, reference_k: int) -> None:
    sub = best_per_fold.dropna(subset=["delta_vs_reference"]).sort_values("delta_vs_reference", ascending=False)
    if len(sub) == 0:
        return
    colors_map = _genogroup_colors(list(sub["genogroup"]))
    colors = [colors_map[g] for g in sub["genogroup"]]

    fig, ax = plt.subplots(figsize=(max(8, 0.18 * len(sub)), 5), dpi=150)
    ax.bar(range(len(sub)), sub["delta_vs_reference"], color=colors)
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub["fold"], rotation=90, fontsize=7)
    ax.set_ylabel(f"ΔAUROC (best k − k={reference_k})")
    ax.set_title(f"Tuning headroom per fold (positive = best-k > k={reference_k})")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.text(0.5, -0.08, _DIAGNOSTIC_FOOTNOTE, ha="center", fontsize=8, style="italic", color="#444")
    _save_both(fig, out_stem)


_CEILING_FOOTNOTE = (
    "Ceiling: k and threshold both tuned per serotype on the held-out novels "
    "(optimistic, NOT deployable). Balanced accuracy = (TPR+TNR)/2; 0.5 = chance. "
    "* n_novel<5 (unreliable)."
)


def _bar_plot_per_serotype(
    df: pd.DataFrame,
    out_stem: Path,
    *,
    value_col: str,
    title: str,
    ylabel: str,
    footnote: str,
    k_col: str | None = None,
) -> None:
    """Sorted per-serotype balanced-accuracy bars, colored by genogroup, with a
    chance line at 0.5 and ``*`` on serotypes with n_novel<5. If ``k_col`` is
    given, the k value is printed above each bar (used for the tuned ceiling)."""
    required = {value_col, "genogroup", "n_novel"} | ({k_col} if k_col else set())
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"best_per_fold is missing {sorted(missing)} — re-run the k-sweep so the "
            "balanced-accuracy / deployment columns are computed."
        )

    sub = df.dropna(subset=[value_col]).sort_values(value_col, ascending=False).reset_index(drop=True)
    colors_map = _genogroup_colors(list(sub["genogroup"]))
    colors = [colors_map[g] for g in sub["genogroup"]]
    labels = [f"{f} *" if n < 5 else str(f) for f, n in zip(sub["fold"], sub["n_novel"])]

    fig, ax = plt.subplots(figsize=(max(8, 0.20 * len(sub)), 5), dpi=150)
    ax.bar(range(len(sub)), sub[value_col], color=colors)
    ax.axhline(0.5, color="#a00000", linewidth=0.9, linestyle="--", label="chance (0.5)")
    if k_col is not None:
        for i, (v, k) in enumerate(zip(sub[value_col], sub[k_col])):
            ax.text(i, min(v + 0.006, 1.04), f"k={int(k)}", rotation=90,
                    ha="center", va="bottom", fontsize=6, color="#333333")
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylim(0.4, 1.08)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.text(0.5, -0.09, footnote, ha="center", fontsize=8, style="italic", color="#444")
    _save_both(fig, out_stem)


def _plot_ceiling_accuracy(best_per_fold: pd.DataFrame, out_stem: Path) -> None:
    """Ceiling: bar = best balanced accuracy at the optimal (k, threshold);
    text above each bar = the optimal k."""
    _bar_plot_per_serotype(
        best_per_fold, out_stem,
        value_col="best_bal_acc", k_col="best_bal_acc_k",
        title="Per-serotype novelty-detection ceiling — balanced accuracy at optimal k & threshold",
        ylabel="Best balanced accuracy (optimal threshold)",
        footnote=_CEILING_FOOTNOTE,
    )


def _plot_deploy_accuracy(best_per_fold: pd.DataFrame, out_stem: Path,
                          deploy_k: int, percentile: float) -> None:
    """Deployment: bar = balanced accuracy at fixed k and a p{percentile}-of-ID
    threshold (set without the novels — honest/realizable)."""
    footnote = (
        f"Deployment: fixed k={deploy_k}, threshold = p{percentile:g} of ID (train LOO) "
        f"k-th-NN distances, set WITHOUT the novels (honest). TNR≈{percentile:g}% by "
        "construction, so bars track detection rate (TPR). Balanced accuracy = "
        "(TPR+TNR)/2; 0.5 = chance. * n_novel<5 (unreliable)."
    )
    _bar_plot_per_serotype(
        best_per_fold, out_stem,
        value_col="deploy_bal_acc", k_col=None,
        title=f"Per-serotype novelty detection — DEPLOYMENT (k={deploy_k}, threshold = p{percentile:g} of ID)",
        ylabel="Balanced accuracy (deployment operating point)",
        footnote=footnote,
    )


# ──────────────────────────── CLI ────────────────────────────


def _parse_k_grid(s: str) -> list[int]:
    return sorted({int(x) for x in s.split(",") if x.strip()})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--loo_dir", required=True,
                   help="Root directory; one subdir per held-out serotype")
    p.add_argument("--output_dir", required=True, help="Where to write CSVs + plots")
    p.add_argument("--k_grid", type=_parse_k_grid,
                   default=list(DEFAULT_K_GRID),
                   help="Comma-separated list of k values to sweep "
                        f"(default: {','.join(str(k) for k in DEFAULT_K_GRID)})")
    p.add_argument(" ", type=int, default=DEFAULT_REFERENCE_K,
                   help="Reference k for the 'tuning headroom' diagnostic "
                        f"(default {DEFAULT_REFERENCE_K}; should be in --k_grid)")
    p.add_argument("--deploy_k", type=int, default=DEFAULT_DEPLOY_K,
                   help="Fixed k for the deployment threshold-accuracy plot "
                        f"(default {DEFAULT_DEPLOY_K}; added to --k_grid if absent)")
    p.add_argument("--deploy_threshold_percentile", type=float, default=DEFAULT_DEPLOY_PERCENTILE,
                   help="Percentile of ID (train LOO) k-th-NN distances used as the "
                        f"deployment novelty threshold (default {DEFAULT_DEPLOY_PERCENTILE}; "
                        "matches knn_ood.py)")
    p.add_argument("--distance_metric", default="cosine", choices=["cosine", "euclidean"])
    p.add_argument("--fold_glob", default="*",
                   help="Glob pattern for fold subdir names (e.g. '6*' to subset)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    loo_dir = Path(args.loo_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for special, name in ((args.reference_k, "--reference_k"), (args.deploy_k, "--deploy_k")):
        if special not in args.k_grid:
            logger.warning("%s=%d not in --k_grid; adding it.", name, special)
            args.k_grid = sorted(set(args.k_grid) | {special})

    fold_dirs = sorted(d for d in loo_dir.glob(args.fold_glob) if d.is_dir())
    logger.info("Found %d candidate fold directories under %s; k grid = %s",
                len(fold_dirs), loo_dir, args.k_grid)

    per_fold_dfs: list[pd.DataFrame] = []
    deploy_records: list[dict] = []
    for fold_dir in fold_dirs:
        result = _sweep_one_fold(fold_dir, args.k_grid, args.distance_metric,
                                 args.deploy_k, args.deploy_threshold_percentile)
        if result is not None:
            df, deploy = result
            per_fold_dfs.append(df)
            deploy_records.append(deploy)

    if not per_fold_dfs:
        logger.error("No usable folds — aborting.")
        return

    per_fold = pd.concat(per_fold_dfs, ignore_index=True)
    summary = _summary_per_k(per_fold)
    best = _best_per_fold(per_fold, args.reference_k)
    # Merge the fixed-k deployment operating point onto the one-row-per-fold table.
    best = best.merge(pd.DataFrame(deploy_records), on="fold", how="left")

    per_fold.to_csv(output_dir / "k_sweep_per_fold.csv", index=False)
    summary.to_csv(output_dir / "k_sweep_summary.csv", index=False)
    best.to_csv(output_dir / "k_sweep_best_per_fold.csv", index=False)

    _plot_per_fold_spaghetti(per_fold, output_dir / "auroc_vs_k_per_fold")
    _plot_aggregated(summary, per_fold, output_dir / "auroc_vs_k_aggregated")
    _plot_best_k_distribution(best, args.k_grid, output_dir / "best_k_distribution")
    _plot_tuning_headroom(best, output_dir / "tuning_headroom", args.reference_k)
    _plot_ceiling_accuracy(best, output_dir / "threshold_accuracy_per_serotype")
    _plot_deploy_accuracy(best, output_dir / "threshold_accuracy_per_serotype_deploy",
                          args.deploy_k, args.deploy_threshold_percentile)

    logger.info("Wrote outputs to %s", output_dir)
    logger.info("  k_sweep_per_fold.csv (%d rows)", len(per_fold))
    logger.info("  k_sweep_summary.csv (%d rows)", len(summary))
    logger.info("  k_sweep_best_per_fold.csv (%d rows)", len(best))
    logger.info("  auroc_vs_k_per_fold.{pdf,png}")
    logger.info("  auroc_vs_k_aggregated.{pdf,png}")
    logger.info("  best_k_distribution.{pdf,png}")
    logger.info("  tuning_headroom.{pdf,png}")
    logger.info("  threshold_accuracy_per_serotype.{pdf,png}  (ceiling: optimal k & threshold)")
    logger.info("  threshold_accuracy_per_serotype_deploy.{pdf,png}  (deployment: k=%d, p%g)",
                args.deploy_k, args.deploy_threshold_percentile)

    # Headline diagnostic to the log
    median_delta = float(best["delta_vs_reference"].median())
    n_meaningful = int((best["delta_vs_reference"] > 0.02).sum())
    logger.info("Headline: median tuning headroom = %+.3f AUROC; %d/%d folds gain >0.02 from tuning.",
                median_delta, n_meaningful, len(best))


if __name__ == "__main__":
    main()
