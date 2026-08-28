"""Aggregate leave-one-serotype-out (LOO) novel-detection performance across folds.

Two detectors are compared: **kNN** (deployed) and **energy** (reference baseline).

Per LOO subdirectory (one held-out serotype each), reads:
    serotype_energies.csv     — ID samples with `energy_serotype`        (stage 003)
    knn_id_distances.csv      — ID samples with `knn_distance`           (stage 004-4)
    query_results.csv         — held-out (novel) samples, energy score   (stage 004q)
    knn_query_distances.csv   — held-out (novel) samples, kNN distance   (stage 004q-knn)
    energy_summary.json       — calibrated energy thresholds             (stage 003)

Computes per-fold AUROC / AUPR / FPR@95TPR / TPR@5FPR / detection-rate metrics
for each method and cross-method agreement (Spearman, Cohen's κ, McNemar,
Jaccard, OR-ensemble ΔAUROC), then aggregates across folds.

Outputs (in --output_dir):
    per_fold_metrics.csv      — one row per (fold, method)
    summary.csv               — one row per method, mean/median/IQR across folds
    method_agreement.csv      — one row per fold
    novel_detection_report.md — auto-generated insights
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    roc_auc_score,
    roc_curve,
)

from ..consts import DEFAULT_ENERGY_TEMPERATURE
from ..logging_config import get_logger
from ..utils import map_serotype_to_group

logger = get_logger(__name__)

ENERGY = "energy"
KNN = "knn"
# kNN is the deployed detector; energy is kept as the reference baseline that
# justifies it. OpenMax and (relative-)Mahalanobis were evaluated over 98 LOO
# folds, lost on every fold-level metric, and have been removed from the repo.
METHODS = (ENERGY, KNN)

AUROC_METRICS = ("auroc", "aupr", "fpr_at_95tpr", "tpr_at_target_fpr", "detection_rate_fitted")


# ─────────────────── data loading ───────────────────


@dataclass
class FoldData:
    fold_name: str          # safe name (slashes → dashes)
    serotype: str           # display name (slashes restored)
    genogroup: str
    id_energy: np.ndarray | None              # per-row ID energy (CBL only)
    id_energy_keys: pd.Series | None          # sample_ids aligned with id_energy
    id_knn: np.ndarray | None                 # per-row ID kth-NN distance
    id_knn_keys: pd.Series | None
    novel_energy: np.ndarray                  # per-row novel energy
    novel_knn: np.ndarray | None              # per-row novel kth-NN distance
    novel_is_novel_energy: np.ndarray         # binary flags from query_results
    novel_is_novel_knn: np.ndarray | None
    energy_threshold_fitted: float | None     # threshold from energy_summary.json at requested percentile

    def id_scores(self, method: str) -> np.ndarray | None:
        return {ENERGY: self.id_energy, KNN: self.id_knn}[method]

    def id_keys(self, method: str) -> pd.Series | None:
        return {ENERGY: self.id_energy_keys, KNN: self.id_knn_keys}[method]

    def novel_scores(self, method: str) -> np.ndarray | None:
        return {ENERGY: self.novel_energy, KNN: self.novel_knn}[method]

    def novel_flags(self, method: str) -> np.ndarray | None:
        return {
            ENERGY: self.novel_is_novel_energy,
            KNN: self.novel_is_novel_knn,
        }[method]

    def has_method(self, method: str) -> bool:
        return self.id_scores(method) is not None and self.novel_scores(method) is not None


def _novel_flag_column(df: pd.DataFrame, path: Path) -> np.ndarray:
    """Read the kNN novelty flag, accepting either column name.

    `knn_ood.py` renamed `is_novel` to `is_novel_knn` so that energy and kNN
    flags are unambiguous once they live in different files. Folds scored before
    that rename still carry the old name, so accept both rather than forcing a
    re-run of every fold."""
    for col in ("is_novel_knn", "is_novel"):
        if col in df.columns:
            return df[col].to_numpy(dtype=bool)
    raise KeyError(f"{path} has neither 'is_novel_knn' nor 'is_novel'")


def _restore_serotype(safe_name: str) -> str:
    """Reverse the slash→dash mangling from loo_array.sh:70 (`SAFE_SEROTYPE="${SEROTYPE//\\//-}"`)."""
    return safe_name.replace("-", "/")


def _load_fold(fold_dir: Path, energy_percentile: float) -> FoldData | None:
    """Load all per-fold inputs. Returns None if the fold is unusable (no novel data)."""
    fold_name = fold_dir.name
    serotype = _restore_serotype(fold_name)
    genogroup = map_serotype_to_group(serotype)

    # Required: novel scores
    query_path = fold_dir / "query_results.csv"
    if not query_path.exists():
        logger.warning("Fold %s: missing query_results.csv — skipping", fold_name)
        return None
    query_df = pd.read_csv(query_path, index_col=0)
    if len(query_df) == 0:
        logger.warning("Fold %s: query_results.csv is empty — skipping", fold_name)
        return None

    novel_energy = query_df["novelty_confidence"].to_numpy(dtype=float)
    novel_is_novel_energy = query_df["is_novel_energy"].to_numpy(dtype=bool)

    # KNN novel scores (separate CSV, since process_trihead_query.py doesn't emit them)
    knn_query_path = fold_dir / "knn_query_distances.csv"
    if knn_query_path.exists():
        knn_query_df = pd.read_csv(knn_query_path)
        # Align to query_df row order via sample_id if possible; else trust positional.
        if "sample_id" in knn_query_df.columns and len(knn_query_df) == len(query_df):
            knn_query_df = knn_query_df.set_index("sample_id").reindex(query_df.index)
        novel_knn = knn_query_df["knn_distance"].to_numpy(dtype=float)
        novel_is_novel_knn = _novel_flag_column(knn_query_df, knn_query_path)
    else:
        novel_knn = None
        novel_is_novel_knn = None

    # Required: ID energies
    energy_id_path = fold_dir / "serotype_energies.csv"
    if not energy_id_path.exists():
        logger.warning("Fold %s: missing serotype_energies.csv — skipping", fold_name)
        return None
    energy_id_df = pd.read_csv(energy_id_path)
    energy_id_df = energy_id_df[energy_id_df["Is_capsule"].astype(bool)]
    id_energy = energy_id_df["energy_serotype"].to_numpy(dtype=float)
    id_energy_keys = energy_id_df["sample_id"].astype(str).reset_index(drop=True)

    # Optional: ID KNN (added by stage 004-4 of loo_array.sh)
    knn_id_path = fold_dir / "knn_id_distances.csv"
    if knn_id_path.exists():
        knn_id_df = pd.read_csv(knn_id_path)
        knn_id_df = knn_id_df[knn_id_df["Is_capsule"].astype(bool)]
        id_knn = knn_id_df["knn_distance"].to_numpy(dtype=float)
        id_knn_keys = knn_id_df["sample_id"].astype(str).reset_index(drop=True)
    else:
        logger.warning(
            "Fold %s: missing knn_id_distances.csv — KNN AUROC will be NaN. "
            "Add stage 004-4 (RUN_KNN_PREDICT_ID) to loo_array.sh.",
            fold_name,
        )
        id_knn = None
        id_knn_keys = None

    # Energy threshold from summary
    energy_threshold_fitted: float | None = None
    summary_path = fold_dir / "energy_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        pct_str = f"{energy_percentile:.1f}"
        percentiles = summary.get("percentiles_serotype", {})
        if pct_str in percentiles:
            energy_threshold_fitted = float(percentiles[pct_str])
        else:
            logger.warning(
                "Fold %s: percentile %s not in energy_summary.json (have %s)",
                fold_name, pct_str, sorted(percentiles.keys()),
            )
        observed_T = summary.get("temperature")
        if observed_T is not None and observed_T != DEFAULT_ENERGY_TEMPERATURE:
            logger.info("Fold %s: energy temperature in summary = %s", fold_name, observed_T)

    return FoldData(
        fold_name=fold_name,
        serotype=serotype,
        genogroup=genogroup,
        id_energy=id_energy,
        id_energy_keys=id_energy_keys,
        id_knn=id_knn,
        id_knn_keys=id_knn_keys,
        novel_energy=novel_energy,
        novel_knn=novel_knn,
        novel_is_novel_energy=novel_is_novel_energy,
        novel_is_novel_knn=novel_is_novel_knn,
        energy_threshold_fitted=energy_threshold_fitted,
    )


# ─────────────────── ROC-family metrics ───────────────────


def _compute_roc_metrics(
    id_scores: np.ndarray,
    novel_scores: np.ndarray,
    fpr_target: float = 0.05,
    bootstrap_n: int = 200,
    rng_seed: int = 42,
) -> dict:
    """Compute AUROC (with bootstrap CI), AUPR, FPR@95TPR, and the operating
    point at a user-specified FPR target (default 5%): TPR + the score
    threshold + the actual achieved FPR. Higher score ⇒ more novel."""
    y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(novel_scores))])
    scores = np.concatenate([id_scores, novel_scores])

    auroc = float(roc_auc_score(y_true, scores))
    aupr = float(average_precision_score(y_true, scores))

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    # FPR @ 95% TPR: smallest FPR at which TPR ≥ 0.95
    idx95 = np.searchsorted(tpr, 0.95)
    fpr_at_95 = float(fpr[idx95]) if idx95 < len(fpr) else float("nan")

    # Operating point at the requested FPR target. We pick the largest TPR
    # among threshold candidates that keep FPR ≤ target. If no candidate
    # satisfies the constraint (only possible when the smallest FPR > target,
    # which happens when novel ≈ ID), we report TPR = 0 at the strictest
    # threshold.
    valid = fpr <= fpr_target
    if valid.any():
        # argmax over the valid region; ties broken by highest threshold
        best_idx = int(np.argmax(np.where(valid, tpr, -1)))
        tpr_at_target = float(tpr[best_idx])
        threshold_at_target = float(thresholds[best_idx])
        achieved_fpr = float(fpr[best_idx])
    else:
        tpr_at_target = 0.0
        threshold_at_target = float(thresholds[0])
        achieved_fpr = float(fpr[0])

    # Bootstrap AUROC CI (resample positive and negative pools independently)
    rng = np.random.default_rng(rng_seed)
    n_id, n_novel = len(id_scores), len(novel_scores)
    boots = np.empty(bootstrap_n, dtype=float)
    for b in range(bootstrap_n):
        ids_b = rng.choice(id_scores, size=n_id, replace=True)
        novel_b = rng.choice(novel_scores, size=n_novel, replace=True)
        y_b = np.concatenate([np.zeros(n_id), np.ones(n_novel)])
        s_b = np.concatenate([ids_b, novel_b])
        boots[b] = roc_auc_score(y_b, s_b)
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])

    return {
        "auroc": auroc,
        "auroc_ci_lo": float(ci_lo),
        "auroc_ci_hi": float(ci_hi),
        "aupr": aupr,
        "fpr_at_95tpr": fpr_at_95,
        "tpr_at_target_fpr": tpr_at_target,
        "threshold_at_target_fpr": threshold_at_target,
        "achieved_fpr": achieved_fpr,
    }


def _detection_rate_at_threshold(scores: np.ndarray, threshold: float) -> float:
    """Fraction of novel samples scored above the fitted ID-percentile threshold."""
    return float((scores > threshold).mean())


def _novel_score_distribution(scores: np.ndarray) -> dict:
    return {
        "novel_score_mean": float(np.mean(scores)),
        "novel_score_std": float(np.std(scores)),
        "novel_score_p05": float(np.percentile(scores, 5)),
        "novel_score_p50": float(np.percentile(scores, 50)),
        "novel_score_p95": float(np.percentile(scores, 95)),
    }


# ─────────────────── method-agreement metrics ───────────────────


def _mcnemar_chi2(flag_a: np.ndarray, flag_b: np.ndarray) -> tuple[float, float]:
    """Discordant-pair χ² with continuity correction. Returns (chi2, two-sided p-value)."""
    from scipy.stats import chi2 as chi2_dist
    b = int(((~flag_a) & flag_b).sum())  # only B detects
    c = int((flag_a & (~flag_b)).sum())  # only A detects
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = float(1 - chi2_dist.cdf(chi2, df=1))
    return float(chi2), p


def _jaccard(flag_a: np.ndarray, flag_b: np.ndarray) -> float:
    inter = int((flag_a & flag_b).sum())
    union = int((flag_a | flag_b).sum())
    return float(inter / union) if union > 0 else float("nan")


def _public_id_means(keys: pd.Series, scores: np.ndarray) -> pd.Series:
    """Group per-contig scores by Public_ID (strip `#Contig_ID` suffix if present) and average."""
    pid = keys.str.split("#", n=1).str[0]
    return pd.DataFrame({"pid": pid, "score": scores}).groupby("pid")["score"].mean()


def _join_two_id_methods(fold: FoldData, m_a: str, m_b: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-Public_ID inner-join of two methods' ID scores. Returns aligned arrays or None."""
    if not (fold.has_method(m_a) and fold.has_method(m_b)):
        return None
    a_means = _public_id_means(fold.id_keys(m_a), fold.id_scores(m_a))
    b_means = _public_id_means(fold.id_keys(m_b), fold.id_scores(m_b))
    common = a_means.index.intersection(b_means.index)
    if len(common) == 0:
        return None
    return a_means.loc[common].to_numpy(), b_means.loc[common].to_numpy()


def _zscore_using(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    mu, sigma = float(reference.mean()), float(reference.std())
    if sigma == 0:
        return target - mu
    return (target - mu) / sigma


def _or_ensemble_auroc(fold: FoldData, m_a: str, m_b: str) -> tuple[float | None, float | None]:
    """AUROC of max(z(method_a), z(method_b)) at Public_ID granularity; delta vs better single."""
    joined = _join_two_id_methods(fold, m_a, m_b)
    if joined is None:
        return None, None
    a_id, b_id = joined
    novel_a = fold.novel_scores(m_a)
    novel_b = fold.novel_scores(m_b)
    a_id_z, b_id_z = _zscore_using(a_id, a_id), _zscore_using(b_id, b_id)
    a_nv_z, b_nv_z = _zscore_using(a_id, novel_a), _zscore_using(b_id, novel_b)

    id_combined = np.maximum(a_id_z, b_id_z)
    novel_combined = np.maximum(a_nv_z, b_nv_z)
    y = np.concatenate([np.zeros(len(id_combined)), np.ones(len(novel_combined))])
    s = np.concatenate([id_combined, novel_combined])
    ens_auroc = float(roc_auc_score(y, s))

    def _single_auroc(id_scores, novel_scores):
        y = np.concatenate([np.zeros(len(id_scores)), np.ones(len(novel_scores))])
        s = np.concatenate([id_scores, novel_scores])
        return float(roc_auc_score(y, s))

    a_auroc = _single_auroc(a_id, novel_a)
    b_auroc = _single_auroc(b_id, novel_b)
    return ens_auroc, float(ens_auroc - max(a_auroc, b_auroc))


def _method_agreement(fold: FoldData) -> list[dict]:
    """Pairwise method-agreement metrics. One row per (method_a, method_b) pair
    where both methods have data for this fold. Computed on the novel set
    (per-contig alignment is trivial inside query_results.csv) plus a
    Public_ID-aggregated Spearman on ID for context."""
    rows: list[dict] = []
    available = [m for m in METHODS if fold.has_method(m)]
    for i, m_a in enumerate(available):
        for m_b in available[i + 1:]:
            scores_a, scores_b = fold.novel_scores(m_a), fold.novel_scores(m_b)
            flags_a, flags_b = fold.novel_flags(m_a), fold.novel_flags(m_b)
            row = {
                "fold": fold.serotype,
                "method_a": m_a,
                "method_b": m_b,
                "n_novel": int(len(scores_a)),
                "spearman_novel": float("nan"),
                "spearman_id_aggregated": float("nan"),
                "cohens_kappa_novel": float("nan"),
                "mcnemar_chi2_novel": float("nan"),
                "mcnemar_pvalue_novel": float("nan"),
                "jaccard_detected_novel": float("nan"),
                "or_ensemble_auroc": float("nan"),
                "delta_auroc_vs_best": float("nan"),
            }
            rho_novel, _ = spearmanr(scores_a, scores_b)
            row["spearman_novel"] = float(rho_novel)
            if flags_a is not None and flags_b is not None:
                row["cohens_kappa_novel"] = float(cohen_kappa_score(flags_a, flags_b))
                chi2, p = _mcnemar_chi2(flags_a, flags_b)
                row["mcnemar_chi2_novel"] = chi2
                row["mcnemar_pvalue_novel"] = p
                row["jaccard_detected_novel"] = _jaccard(flags_a, flags_b)
            joined_id = _join_two_id_methods(fold, m_a, m_b)
            if joined_id is not None:
                rho_id, _ = spearmanr(joined_id[0], joined_id[1])
                row["spearman_id_aggregated"] = float(rho_id)
            ens, delta = _or_ensemble_auroc(fold, m_a, m_b)
            if ens is not None:
                row["or_ensemble_auroc"] = ens
                row["delta_auroc_vs_best"] = delta
            rows.append(row)
    return rows


# ─────────────────── per-fold orchestration ───────────────────


_NAN_METRIC_KEYS = (
    "auroc", "auroc_ci_lo", "auroc_ci_hi", "aupr",
    "fpr_at_95tpr", "tpr_at_target_fpr",
    "threshold_at_target_fpr", "achieved_fpr",
    "novel_score_mean", "novel_score_std",
    "novel_score_p05", "novel_score_p50", "novel_score_p95",
    "detection_rate_fitted", "fitted_threshold",
)


def _per_fold_method_metrics(fold: FoldData, bootstrap_n: int, fpr_target: float) -> list[dict]:
    """One row per (fold, method). Methods without ID scores get NaN-filled rows."""
    rows: list[dict] = []
    for method in METHODS:
        id_scores = fold.id_scores(method)
        novel_scores = fold.novel_scores(method)
        row = {
            "fold": fold.serotype,
            "fold_safe_name": fold.fold_name,
            "genogroup": fold.genogroup,
            "method": method,
            "n_id": int(len(id_scores)) if id_scores is not None else 0,
            "n_novel": int(len(novel_scores)) if novel_scores is not None else 0,
        }
        if id_scores is not None and novel_scores is not None:
            row.update(_compute_roc_metrics(id_scores, novel_scores, fpr_target, bootstrap_n))
            row.update(_novel_score_distribution(novel_scores))
            if method == ENERGY and fold.energy_threshold_fitted is not None:
                row["detection_rate_fitted"] = _detection_rate_at_threshold(
                    novel_scores, fold.energy_threshold_fitted
                )
                row["fitted_threshold"] = fold.energy_threshold_fitted
            else:
                # KNN: use the binary flag from the predict step.
                flags = fold.novel_flags(method)
                row["detection_rate_fitted"] = float(flags.mean()) if flags is not None else float("nan")
                row["fitted_threshold"] = float("nan")
        else:
            for k in _NAN_METRIC_KEYS:
                row[k] = float("nan")
            flags = fold.novel_flags(method)
            if flags is not None:
                row["detection_rate_fitted"] = float(flags.mean())
        rows.append(row)
    return rows


# ─────────────────── cross-fold summary ───────────────────


def _summary_table(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Mean / median / IQR / min / max per (method × metric)."""
    metric_cols = [
        "auroc", "aupr", "fpr_at_95tpr", "tpr_at_target_fpr",
        "detection_rate_fitted", "novel_score_mean",
    ]
    rows = []
    for method, sub in per_fold.groupby("method"):
        for col in metric_cols:
            vals = sub[col].dropna().to_numpy()
            if len(vals) == 0:
                rows.append({"method": method, "metric": col, "n_folds": 0,
                             "mean": float("nan"), "median": float("nan"),
                             "iqr": float("nan"), "std": float("nan"),
                             "min": float("nan"), "max": float("nan")})
                continue
            q25, q75 = np.percentile(vals, [25, 75])
            rows.append({
                "method": method, "metric": col, "n_folds": len(vals),
                "mean": float(vals.mean()), "median": float(np.median(vals)),
                "iqr": float(q75 - q25), "std": float(vals.std()),
                "min": float(vals.min()), "max": float(vals.max()),
            })
    # Win-counts: per fold, which method has the highest AUROC?
    pivot = per_fold.pivot_table(index="fold", columns="method", values="auroc")
    valid = pivot.dropna(how="all")
    if len(valid) > 0:
        wins = {m: 0 for m in METHODS}
        for fold, row in valid.iterrows():
            present = row.dropna()
            if len(present) == 0:
                continue
            best = present.idxmax()
            wins[best] += 1
        for m, w in wins.items():
            rows.append({"method": m, "metric": "auroc_win_count",
                         "n_folds": len(valid), "mean": float(w),
                         "median": float("nan"), "iqr": float("nan"),
                         "std": float("nan"), "min": float("nan"), "max": float(w)})
    return pd.DataFrame(rows)


def _per_genogroup_summary(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Long-format per-genogroup summary (one row per method × genogroup)."""
    rows = []
    for (method, genogroup), sub in per_fold.groupby(["method", "genogroup"]):
        auroc_vals = sub["auroc"].dropna().to_numpy()
        tpr_vals = sub["tpr_at_target_fpr"].dropna().to_numpy()
        if len(auroc_vals) == 0:
            continue
        rows.append({
            "method": method, "genogroup": genogroup,
            "n_folds": len(auroc_vals),
            "auroc_mean": float(auroc_vals.mean()),
            "auroc_min": float(auroc_vals.min()),
            "auroc_max": float(auroc_vals.max()),
            "tpr_at_target_mean": float(tpr_vals.mean()) if len(tpr_vals) else float("nan"),
            "tpr_at_target_min": float(tpr_vals.min()) if len(tpr_vals) else float("nan"),
            "tpr_at_target_max": float(tpr_vals.max()) if len(tpr_vals) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values(["method", "auroc_mean"])


def _per_genogroup_wide(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Wide-format per-genogroup table: one row per genogroup, all methods side
    by side. Each method gets a mean AUROC and mean TPR@target_fpr column."""
    # n_folds is consistent across methods within a genogroup (same fold set)
    n_folds = (per_fold[per_fold["method"] == ENERGY]
               .groupby("genogroup")["fold"].nunique()
               .rename("n_folds"))
    pivots = []
    for metric, short in [("auroc", "auroc"), ("tpr_at_target_fpr", "tpr")]:
        p = (per_fold.pivot_table(index="genogroup", columns="method",
                                   values=metric, aggfunc="mean"))
        p.columns = [f"{short}_{m}" for m in p.columns]
        pivots.append(p)
    out = pd.concat([n_folds] + pivots, axis=1).reset_index()
    # Sort by the average of all methods' AUROC ascending — hardest at top
    auroc_cols = [c for c in out.columns if c.startswith("auroc_")]
    out["_sort_key"] = out[auroc_cols].mean(axis=1)
    out = out.sort_values("_sort_key").drop(columns="_sort_key")
    return out


def _auroc_summary(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Slim per-method AUROC summary (one row per method). Quantile-rich for
    skewed distributions; medians and IQR are more informative than mean ± std."""
    rows = []
    for method, sub in per_fold.groupby("method"):
        vals = sub["auroc"].dropna().to_numpy()
        if len(vals) == 0:
            continue
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        rows.append({
            "method": method,
            "n_folds": len(vals),
            "mean": float(vals.mean()),
            "median": float(q50),
            "p25": float(q25),
            "p75": float(q75),
            "iqr": float(q75 - q25),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
        })
    out = pd.DataFrame(rows)
    # Annotate AUROC-win counts (per-fold argmax over the methods that have a value)
    pivot = per_fold.pivot_table(index="fold", columns="method", values="auroc")
    wins = pivot.apply(lambda r: r.dropna().idxmax() if r.dropna().any() else None, axis=1).value_counts()
    out["wins"] = out["method"].map(wins).fillna(0).astype(int)
    return out.sort_values("median", ascending=False)


def _tpr_at_fpr_summary(per_fold: pd.DataFrame, fpr_target: float) -> pd.DataFrame:
    """Per-method TPR@target_fpr summary. For deployment, the operationally
    meaningful single number per method."""
    rows = []
    for method, sub in per_fold.groupby("method"):
        vals = sub["tpr_at_target_fpr"].dropna().to_numpy()
        if len(vals) == 0:
            continue
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        # Achieved FPR is whatever the ROC curve actually hits closest to the
        # target (constrained to ≤ target); useful as a sanity check.
        ach = sub["achieved_fpr"].dropna().to_numpy()
        rows.append({
            "method": method,
            "fpr_target": float(fpr_target),
            "n_folds": len(vals),
            "tpr_mean": float(vals.mean()),
            "tpr_median": float(q50),
            "tpr_p25": float(q25),
            "tpr_p75": float(q75),
            "tpr_min": float(vals.min()),
            "tpr_max": float(vals.max()),
            "frac_folds_tpr_above_0.5": float((vals > 0.5).mean()),
            "frac_folds_tpr_above_0.8": float((vals > 0.8).mean()),
            "achieved_fpr_mean": float(ach.mean()) if len(ach) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("tpr_median", ascending=False)


# ─────────────────── markdown report ───────────────────


def _fmt(v: float, n: int = 3) -> str:
    return "n/a" if not np.isfinite(v) else f"{v:.{n}f}"


def _df_to_md_table(df: pd.DataFrame, float_fmt: str = ".3f") -> str:
    """Tiny dependency-free markdown table renderer (avoids pandas's tabulate dep)."""
    if len(df) == 0:
        return "_(empty)_\n"
    cols = list(df.columns)
    def _cell(v):
        if isinstance(v, float):
            return "n/a" if not np.isfinite(v) else format(v, float_fmt)
        return str(v)
    rows = [[_cell(v) for v in row] for row in df.to_numpy()]
    widths = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)]
    header = "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = "\n".join("| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(cols))) + " |" for r in rows)
    return "\n".join([header, sep, body]) + "\n"


def _fmt_signed(v: float, n: int = 3) -> str:
    """Format with explicit sign for non-negative values; negative numbers keep their minus sign."""
    if not np.isfinite(v):
        return "n/a"
    return f"+{v:.{n}f}" if v >= 0 else f"{v:.{n}f}"


def _build_report(
    per_fold: pd.DataFrame,
    summary: pd.DataFrame,
    agreement: pd.DataFrame,
    genogroup_summary: pd.DataFrame,
    auroc_summary_df: pd.DataFrame,
    tpr_summary_df: pd.DataFrame,
    genogroup_wide_df: pd.DataFrame,
    energy_percentile: float,
    fpr_target: float,
) -> str:
    n_folds = per_fold["fold"].nunique()
    fpr_pct = fpr_target * 100

    def _row(method: str, metric: str) -> dict | None:
        sub = summary[(summary["method"] == method) & (summary["metric"] == metric)]
        if len(sub) == 0 or not np.isfinite(sub.iloc[0]["mean"]):
            return None
        return sub.iloc[0].to_dict()

    present_methods = [m for m in METHODS if _row(m, "auroc") is not None]

    # ── Headline: slim AUROC table + slim TPR@FPR table side by side
    headline = "## Headline\n\n"
    headline += "**AUROC across folds** (per-method summary):\n\n"
    auroc_show = auroc_summary_df[["method", "n_folds", "mean", "median", "p25", "p75", "min", "max", "wins"]].copy()
    headline += _df_to_md_table(auroc_show) + "\n"
    headline += f"**TPR at FPR = {fpr_pct:.2g}%** (the deployment operating point):\n\n"
    tpr_show = tpr_summary_df[[
        "method", "n_folds", "tpr_mean", "tpr_median", "tpr_p25", "tpr_p75",
        "tpr_min", "tpr_max", "frac_folds_tpr_above_0.5",
    ]].copy()
    headline += _df_to_md_table(tpr_show) + "\n"

    # ── Confidence: cross-fold CI on the mean for each method
    confidence = "## Reading the confidence\n\n"
    if n_folds < 2:
        confidence += (f"_Only {n_folds} fold available — skipping cross-fold "
                       f"confidence intervals (need ≥ 2). Per-fold bootstrap CIs are "
                       f"in `per_fold_metrics.csv`._\n")
    else:
        method_cis: dict[str, tuple[float, float]] = {}
        for m in present_methods:
            a = _row(m, "auroc")
            if a["n_folds"] < 2:
                continue
            se = a["std"] / np.sqrt(a["n_folds"])
            lo, hi = a["mean"] - 1.96 * se, a["mean"] + 1.96 * se
            method_cis[m] = (lo, hi)
            confidence += (f"- {m.capitalize()} AUROC across-fold 95% CI on the mean: "
                           f"[{_fmt(lo)}, {_fmt(hi)}]"
                           f"{' — overlaps 0.5 (no skill)' if lo <= 0.5 <= hi else ''}.\n")
        # Pairwise CI overlap statements
        pairs = [(a, b) for i, a in enumerate(present_methods) for b in present_methods[i + 1:]]
        for a, b in pairs:
            if a not in method_cis or b not in method_cis:
                continue
            a_lo, a_hi = method_cis[a]
            b_lo, b_hi = method_cis[b]
            overlap = not (a_hi < b_lo or b_hi < a_lo)
            confidence += (
                f"- {a.capitalize()} vs {b.capitalize()} CIs {'overlap' if overlap else 'do NOT overlap'}: "
                f"{'NOT statistically distinguishable' if overlap else 'statistically distinguishable'} "
                f"at the fold level.\n"
            )

    # ── Pairwise redundancy/complementarity verdicts
    verdict = "## Pairwise redundancy / complementarity\n\n"
    if len(agreement) == 0:
        verdict += "_No method pairs available (only one method has ID + novel data)._\n"
    else:
        rows = []
        for (a, b), sub in agreement.groupby(["method_a", "method_b"]):
            rho_n = sub["spearman_novel"].dropna()
            rho_id = sub["spearman_id_aggregated"].dropna()
            kappa = sub["cohens_kappa_novel"].dropna()
            delta = sub["delta_auroc_vs_best"].dropna()
            rows.append({
                "pair": f"{a} ↔ {b}",
                "n_folds": len(sub),
                "rho_novel_med": float(rho_n.median()) if len(rho_n) else float("nan"),
                "rho_id_med": float(rho_id.median()) if len(rho_id) else float("nan"),
                "kappa_med": float(kappa.median()) if len(kappa) else float("nan"),
                "OR_ens_ΔAUROC_med": float(delta.median()) if len(delta) else float("nan"),
            })
        verdict += _df_to_md_table(pd.DataFrame(rows)) + "\n"

        # Per-pair verdict uses rho on the novel set rather than on ID, where a
        # shared bulk of near-duplicate loci inflates the correlation.
        for r in rows:
            pair = r["pair"]
            rho = r["rho_novel_med"]
            delta = r["OR_ens_ΔAUROC_med"]
            kappa = r["kappa_med"]
            if not np.isfinite(rho) or not np.isfinite(delta):
                continue
            if delta >= 0.02:
                line = (f"  - **{pair}**: meaningful ensemble gain "
                        f"(ΔAUROC {_fmt_signed(delta)}); keep both, prefer the OR-ensemble.")
            elif delta < -0.005:
                line = (f"  - **{pair}**: OR-ensemble HURTS "
                        f"(ΔAUROC {_fmt_signed(delta)}); likely score-scale mismatch. "
                        f"Use the better single method.")
            elif rho > 0.7 and abs(delta) < 0.01:
                line = (f"  - **{pair}**: redundant (ρ {_fmt(rho)}, ΔAUROC {_fmt_signed(delta)}); "
                        f"either method captures the same signal.")
            elif rho < 0.0:
                line = (f"  - **{pair}**: anti-correlated on novel (ρ {_fmt(rho)}, "
                        f"κ {_fmt(kappa)}) but no ensemble gain (ΔAUROC {_fmt_signed(delta)}) "
                        f"— check the score scales.")
            else:
                line = (f"  - **{pair}**: marginal complementarity (ρ {_fmt(rho)}, "
                        f"ΔAUROC {_fmt_signed(delta)}); keep the better single method as default.")
            verdict += line + "\n"

    # ── Hardest folds (lowest AUROC of the best-available method, with
    # TPR@FPR_target for that method so the operating-point implication is visible)
    hardest = "## Hardest folds (lowest AUROC)\n\n"
    ref_method = ENERGY if ENERGY in present_methods else (present_methods[0] if present_methods else None)
    if ref_method is not None:
        ref_perfold = (per_fold[per_fold["method"] == ref_method]
                       .sort_values("auroc")
                       .head(10)[["fold", "genogroup", "auroc", "aupr",
                                  "tpr_at_target_fpr", "n_novel"]])
        if len(ref_perfold) > 0:
            hardest += f"**By {ref_method} AUROC** (TPR shown at FPR = {fpr_pct:.2g}%):\n\n"
            hardest += ref_perfold.pipe(_df_to_md_table) + "\n\n"

    # ── Per-genogroup breakdown: wide table (one row per genogroup, all methods side by side)
    genogroup_section = f"## Per-genogroup table\n\n_AUROC and TPR@{fpr_pct:.2g}%FPR per method, one row per genogroup, sorted hardest-first._\n\n"
    if len(genogroup_wide_df) > 0:
        genogroup_section += _df_to_md_table(genogroup_wide_df) + "\n"
        ref_long = genogroup_summary[genogroup_summary["method"] == ref_method].sort_values("auroc_mean")
        if len(ref_long) > 1:
            worst = ref_long.iloc[0]
            best = ref_long.iloc[-1]
            spread = float(best["auroc_mean"] - worst["auroc_mean"])
            if spread > 0.1:
                genogroup_section += (
                    f"_Spread of {spread:.2f} AUROC between worst genogroup "
                    f"({worst['genogroup']}) and best ({best['genogroup']}) for {ref_method} — "
                    f"strong evidence that near-OOD (sibling-serotype) cases drive failures._\n\n"
                )

    # ── Operating-point recommendation: per-method TPR@target with per-fold thresholds
    op = f"## Operating point @ FPR = {fpr_pct:.2g}%\n\n"
    op += ("_To deploy, hold out a calibration split of folds, set the threshold "
           "to the median `threshold_at_target_fpr` over those folds for your chosen "
           "method, and apply it in production. Below: per-method achievable TPR "
           "and threshold stability across folds._\n\n")
    for m in present_methods:
        sub = per_fold[per_fold["method"] == m]
        tpr = sub["tpr_at_target_fpr"].dropna()
        thr = sub["threshold_at_target_fpr"].dropna()
        if len(tpr) == 0:
            continue
        op += (f"- **{m.capitalize()}**: TPR median {_fmt(tpr.median())} "
               f"(IQR {_fmt(tpr.quantile(0.25))}–{_fmt(tpr.quantile(0.75))}). ")
        if len(thr) >= 2:
            op += (f"Threshold across folds: median {_fmt(thr.median())} "
                   f"(IQR {_fmt(thr.quantile(0.25))}–{_fmt(thr.quantile(0.75))}, "
                   f"range [{_fmt(thr.min())}, {_fmt(thr.max())}]).")
        op += "\n"
    op += "\n"
    op += "### Legacy: detection rate at the existing percentile-of-ID threshold\n\n"
    e_thr = per_fold[per_fold["method"] == ENERGY]["fitted_threshold"].dropna()
    if len(e_thr) >= 2:
        thr_std = float(e_thr.std())
        op += (f"- {energy_percentile:.1f}th-percentile energy threshold across folds: "
               f"[{_fmt(e_thr.min())}, {_fmt(e_thr.max())}] (std {_fmt(thr_std)}). "
               f"{'Stable.' if thr_std < 0.5 else 'Unstable — percentile threshold drifts across folds.'}\n")
    elif len(e_thr) == 1:
        op += (f"- Single fitted threshold observed: {_fmt(e_thr.iloc[0])} (need ≥ 2 folds).\n")
    for m in present_methods:
        dr = per_fold[per_fold["method"] == m]["detection_rate_fitted"].dropna()
        if len(dr) > 0:
            op += (f"- Detection rate at fitted threshold ({m}): "
                   f"mean {_fmt(dr.mean())} (range [{_fmt(dr.min())}, {_fmt(dr.max())}]).\n")

    return (
        f"# Novel-detection LOO summary\n\n"
        f"_{n_folds} folds aggregated; deployment operating point = {fpr_pct:.2g}% FPR; "
        f"legacy energy percentile = {energy_percentile:.1f}%._\n\n"
        + headline + "\n"
        + confidence + "\n"
        + verdict + "\n"
        + hardest
        + genogroup_section
        + op
    )


# ─────────────────── CLI ───────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--loo_dir", required=True,
                   help="Root directory; one subdir per held-out serotype")
    p.add_argument("--output_dir", required=True, help="Where to write summary outputs")
    p.add_argument("--energy_percentile", type=float, default=99.0,
                   help="Percentile (in energy_summary.json) used as the fitted operating point")
    p.add_argument("--fpr_target", type=float, default=0.05,
                   help="Target FPR for the deployment operating point (default 0.05 = 5%%). "
                        "Per fold and per method we report the TPR achievable at this FPR plus "
                        "the corresponding score threshold.")
    p.add_argument("--bootstrap_n", type=int, default=200,
                   help="Bootstrap resamples for AUROC CI (per fold)")
    p.add_argument("--fold_glob", default="*",
                   help="Glob pattern for fold subdir names (e.g. '6*' to subset)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    loo_dir = Path(args.loo_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_dirs = sorted(d for d in loo_dir.glob(args.fold_glob) if d.is_dir())
    logger.info("Found %d candidate fold directories under %s", len(fold_dirs), loo_dir)

    folds: list[FoldData] = []
    for d in fold_dirs:
        fold = _load_fold(d, args.energy_percentile)
        if fold is not None:
            folds.append(fold)
    logger.info("Loaded %d usable folds", len(folds))
    if not folds:
        logger.error("No usable folds — aborting.")
        return

    per_fold_rows: list[dict] = []
    agreement_rows: list[dict] = []
    for fold in folds:
        per_fold_rows.extend(_per_fold_method_metrics(fold, args.bootstrap_n, args.fpr_target))
        agreement_rows.extend(_method_agreement(fold))

    per_fold_df = pd.DataFrame(per_fold_rows)
    agreement_df = pd.DataFrame(agreement_rows)
    summary_df = _summary_table(per_fold_df)
    genogroup_df = _per_genogroup_summary(per_fold_df)
    genogroup_wide_df = _per_genogroup_wide(per_fold_df)
    auroc_summary_df = _auroc_summary(per_fold_df)
    tpr_summary_df = _tpr_at_fpr_summary(per_fold_df, args.fpr_target)

    per_fold_df.to_csv(output_dir / "per_fold_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    agreement_df.to_csv(output_dir / "method_agreement.csv", index=False)
    genogroup_df.to_csv(output_dir / "per_genogroup_summary.csv", index=False)
    genogroup_wide_df.to_csv(output_dir / "per_genogroup_wide.csv", index=False)
    auroc_summary_df.to_csv(output_dir / "auroc_summary.csv", index=False)
    tpr_summary_df.to_csv(output_dir / "tpr_at_fpr_summary.csv", index=False)

    report = _build_report(
        per_fold_df, summary_df, agreement_df, genogroup_df,
        auroc_summary_df, tpr_summary_df, genogroup_wide_df,
        args.energy_percentile, args.fpr_target,
    )
    (output_dir / "novel_detection_report.md").write_text(report)

    logger.info("Wrote outputs to %s", output_dir)
    logger.info("  per_fold_metrics.csv (%d rows)", len(per_fold_df))
    logger.info("  summary.csv (%d rows)", len(summary_df))
    logger.info("  method_agreement.csv (%d rows)", len(agreement_df))
    logger.info("  per_genogroup_summary.csv (%d rows; long)", len(genogroup_df))
    logger.info("  per_genogroup_wide.csv (%d rows; one row per genogroup)", len(genogroup_wide_df))
    logger.info("  auroc_summary.csv (%d rows; one row per method)", len(auroc_summary_df))
    logger.info("  tpr_at_fpr_summary.csv (%d rows; TPR @ %.2g%% FPR)",
                len(tpr_summary_df), args.fpr_target * 100)
    logger.info("  novel_detection_report.md")


if __name__ == "__main__":
    main()
