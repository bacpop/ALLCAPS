#!/usr/bin/env python
"""Mahalanobis-OOD — class-conditional Gaussian likelihood for novel detection.

Implements Lee et al. NeurIPS 2018 ("A Simple Unified Framework for Detecting
Out-of-Distribution Samples and Adversarial Attacks") and Ren et al. 2021's
relative-Mahalanobis variant in one fit/predict cycle.

Idea
----
Fit one Gaussian per serotype in the trihead embedding space, with a tied
within-class covariance Σ (Ledoit-Wolf shrinkage for stability when classes
are small). Per query x:

  * Per-class squared distance:  M²_s(x) = (x - μ_s)ᵀ Σ⁻¹ (x - μ_s)
  * Vanilla score:               M_min(x)  = √min_s M²_s(x)        (higher = more novel)
  * Relative score:              RM(x)     = M_min(x) − √M²_0(x)   (subtracts background; sharpens near-OOD)
  * Ranked likelihood list:      sort classes by  log p(x|s) ∝ -½ M²_s(x)

Why both vanilla + relative: the relative form subtracts a background term
fit on all classes pooled, isolating the "class-specific" component. Often
helps when the held-out serotype shares the broad geometric region of its
training siblings (the cases where vanilla KNN fails).

Why tied (not per-class) covariance: many serotypes have <50 training samples;
estimating a full 128×128 covariance per class would be singular for most of
them. The tied estimator pools within-class deviations, which gives a stable
shape estimate that's a function of all training data.

Usage
-----
  # Fit:
  python -m scripts.mahalanobis_ood fit \\
      --embeddings inference_results.npz \\
      --labels final_metadata.csv \\
      --output mahalanobis_params.pkl

  # Predict on ID (for cross-method AUROC):
  python -m scripts.mahalanobis_ood predict \\
      --input_type id \\
      --embeddings inference_results.npz \\
      --labels final_metadata.csv \\
      --mahalanobis_params mahalanobis_params.pkl \\
      --output_dir <fold_dir>

  # Predict on query (held-out novel):
  python -m scripts.mahalanobis_ood predict \\
      --input_type query \\
      --embeddings query_embeddings.npz \\
      --mahalanobis_params mahalanobis_params.pkl \\
      --output_dir <fold_dir>
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from sklearn.covariance import LedoitWolf

from .consts import DEFAULT_MISSING_LABEL, DEFAULT_SEP
from .knn_ood import _load_id_embeddings, _load_query_embeddings
from .logging_config import get_logger
from .utils import get_sample_id

logger = get_logger(__name__)

DEFAULT_TOPK = 5


# ──────────────────────────── core ────────────────────────────


class MahalanobisOOD:
    """Per-class Gaussian discriminant model with tied covariance + a
    background Gaussian, optimized for both classification and OOD scoring."""

    def __init__(self, shrinkage: str = "lw"):
        if shrinkage not in ("lw", "none"):
            raise ValueError(f"shrinkage must be 'lw' or 'none', got {shrinkage}")
        self.shrinkage = shrinkage
        # Fitted state:
        self.class_means: np.ndarray | None = None         # (S, D)
        self.idx_to_class: dict[int, str] | None = None
        self.L_inv: np.ndarray | None = None               # (D, D) lower-triangular inverse, so M²_s(x) = ‖L⁻¹(x-μ_s)‖²
        self.bg_mean: np.ndarray | None = None             # (D,)
        self.L_inv_bg: np.ndarray | None = None
        # Calibrated training-set scores for threshold derivation:
        self.train_mahalanobis_scores: np.ndarray | None = None
        self.train_relative_mahalanobis_scores: np.ndarray | None = None

    # ─────────── fit ───────────

    def _fit_inv_cholesky(self, cov: np.ndarray) -> np.ndarray:
        """Return L⁻¹ where L is the lower-triangular Cholesky factor of cov."""
        # cov is symmetric PD (LedoitWolf guarantees this). Use scipy's cho_factor.
        c, lower = cho_factor(cov, lower=True)
        L = np.tril(c)  # cho_factor returns the factor in-place; lower-tri view
        # L_inv via solve_triangular with identity
        L_inv = solve_triangular(L, np.eye(cov.shape[0]), lower=True)
        return L_inv

    def fit(self, X: np.ndarray, labels: np.ndarray) -> None:
        """X: (N, D); labels: (N,) string serotype labels."""
        assert X.ndim == 2 and X.shape[0] == len(labels), "X / labels mismatch"
        classes = sorted(set(labels))
        S, D = len(classes), X.shape[1]
        class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in class_to_idx.items()}

        # Per-class means
        means = np.zeros((S, D), dtype=np.float64)
        n_per_class = np.zeros(S, dtype=np.int64)
        for s, c in enumerate(classes):
            mask = labels == c
            means[s] = X[mask].mean(axis=0)
            n_per_class[s] = int(mask.sum())
        self.class_means = means.astype(np.float32)

        # Pooled within-class deviations → tied Σ
        centered = X - means[np.array([class_to_idx[lbl] for lbl in labels])]
        if self.shrinkage == "lw":
            cov_est = LedoitWolf().fit(centered)
            cov = cov_est.covariance_
            logger.info("Tied covariance (Ledoit-Wolf): shrinkage=%.4f", cov_est.shrinkage_)
        else:
            cov = np.cov(centered.T, ddof=1)
        # Numerical safety: regularize the diagonal a touch
        cov = cov + 1e-6 * np.eye(D)
        self.L_inv = self._fit_inv_cholesky(cov).astype(np.float32)

        # Background Gaussian on all training samples (pooled)
        self.bg_mean = X.mean(axis=0).astype(np.float32)
        if self.shrinkage == "lw":
            bg_cov = LedoitWolf().fit(X).covariance_
        else:
            bg_cov = np.cov(X.T, ddof=1)
        bg_cov = bg_cov + 1e-6 * np.eye(D)
        self.L_inv_bg = self._fit_inv_cholesky(bg_cov).astype(np.float32)

        # Calibrate training-set scores (used to derive percentile-based threshold
        # at predict time, matching KNN's convention).
        mahal, rel_mahal, _ = self._score_internal(X.astype(np.float32), topk=0)
        self.train_mahalanobis_scores = mahal.astype(np.float32)
        self.train_relative_mahalanobis_scores = rel_mahal.astype(np.float32)

        logger.info(
            "Mahalanobis fit: n=%d, S=%d classes, D=%d. "
            "Class sizes: min=%d, median=%d, max=%d. "
            "Train M_min: median=%.3f, p99=%.3f. Train RM: median=%.3f, p99=%.3f.",
            X.shape[0], S, D, int(n_per_class.min()),
            int(np.median(n_per_class)), int(n_per_class.max()),
            float(np.median(self.train_mahalanobis_scores)),
            float(np.percentile(self.train_mahalanobis_scores, 99)),
            float(np.median(self.train_relative_mahalanobis_scores)),
            float(np.percentile(self.train_relative_mahalanobis_scores, 99)),
        )

    # ─────────── score ───────────

    def _score_internal(
        self,
        X: np.ndarray,
        topk: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Return (M_min, RM, per_class_M²) where per_class_M² is (N, S) or
        None if topk==0 (memory saver for training-set calibration)."""
        # Whitened query vs each class: M²_s(x) = ‖L⁻¹ (x - μ_s)‖²
        # Compute L⁻¹ X.T  and  L⁻¹ μ.T  once each, then squared L2 differences.
        # (N, D) @ (D, D).T = (N, D) — but L_inv is lower-tri (D, D)
        Xw = X @ self.L_inv.T          # (N, D)
        Mw = self.class_means @ self.L_inv.T  # (S, D)
        # Squared distance matrix (N, S)
        x_sq = (Xw * Xw).sum(axis=1, keepdims=True)         # (N, 1)
        m_sq = (Mw * Mw).sum(axis=1)                          # (S,)
        cross = Xw @ Mw.T                                     # (N, S)
        D2 = x_sq + m_sq[None, :] - 2.0 * cross
        np.maximum(D2, 0, out=D2)  # numerical clamp
        M_min2 = D2.min(axis=1)
        M_min = np.sqrt(M_min2)

        # Background squared Mahalanobis
        Xb = (X - self.bg_mean) @ self.L_inv_bg.T  # (N, D)
        M_bg2 = (Xb * Xb).sum(axis=1)
        M_bg = np.sqrt(np.maximum(M_bg2, 0))
        RM = M_min - M_bg

        D2_out = D2 if topk > 0 else None
        return M_min.astype(np.float32), RM.astype(np.float32), D2_out

    def score(self, X: np.ndarray, topk: int = DEFAULT_TOPK) -> dict:
        """Returns dict with:
            mahalanobis_dist:        (N,) M_min(x)
            relative_mahalanobis_dist: (N,) RM(x)
            ranked: list of length N; each entry a list of {class, dist, log_lik, softmax_prob}
                    sorted ascending by distance (most-likely class first), length topk.
        """
        assert self.L_inv is not None, "Call fit() first."
        X = np.asarray(X, dtype=np.float32)
        M_min, RM, D2 = self._score_internal(X, topk=topk)

        ranked: list[list[dict]] = []
        if topk > 0 and D2 is not None:
            S = D2.shape[1]
            k = min(topk, S)
            # Argpartition for the k smallest distances per query, then sort that subset
            partial_idx = np.argpartition(D2, kth=k - 1, axis=1)[:, :k]
            for i in range(len(X)):
                idxs = partial_idx[i]
                order = idxs[np.argsort(D2[i, idxs])]
                # log p(x|s) ∝ -½ M²_s. Softmax over -½ M²_s gives a normalized posterior
                # (under a flat class prior, which matches our LDA-style fit).
                log_liks = -0.5 * D2[i, order]
                # Numerically stable softmax over the top-K (not full S — this is a
                # local confidence, normalised over the top-K only).
                log_liks_shifted = log_liks - log_liks.max()
                exps = np.exp(log_liks_shifted)
                probs = exps / exps.sum()
                ranked.append([
                    {
                        "class": self.idx_to_class[int(order[j])],
                        "mahalanobis_dist": float(np.sqrt(D2[i, order[j]])),
                        "log_likelihood": float(log_liks[j]),
                        "softmax_prob_topk": float(probs[j]),
                    }
                    for j in range(k)
                ])
        return {
            "mahalanobis_dist": M_min,
            "relative_mahalanobis_dist": RM,
            "ranked": ranked,
        }

    # ─────────── persistence ───────────

    def save(self, path: str) -> None:
        assert self.class_means is not None, "Cannot save unfitted MahalanobisOOD."
        with open(path, "wb") as f:
            pickle.dump({
                "shrinkage": self.shrinkage,
                "class_means": self.class_means,
                "idx_to_class": self.idx_to_class,
                "L_inv": self.L_inv,
                "bg_mean": self.bg_mean,
                "L_inv_bg": self.L_inv_bg,
                "train_mahalanobis_scores": self.train_mahalanobis_scores,
                "train_relative_mahalanobis_scores": self.train_relative_mahalanobis_scores,
            }, f)
        logger.info("Mahalanobis parameters saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "MahalanobisOOD":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(shrinkage=data["shrinkage"])
        obj.class_means = data["class_means"]
        obj.idx_to_class = data["idx_to_class"]
        obj.L_inv = data["L_inv"]
        obj.bg_mean = data["bg_mean"]
        obj.L_inv_bg = data["L_inv_bg"]
        obj.train_mahalanobis_scores = data["train_mahalanobis_scores"]
        obj.train_relative_mahalanobis_scores = data["train_relative_mahalanobis_scores"]
        return obj


# ──────────────────────────── CLI: fit ────────────────────────────


def cli_fit(args: argparse.Namespace) -> None:
    X, labels_df, _ = _load_id_embeddings(args.embeddings, args.labels, sep=args.sep)
    labels = labels_df["Serotype"].to_numpy()
    mahal = MahalanobisOOD(shrinkage=args.shrinkage)
    mahal.fit(X.astype(np.float32), labels)
    mahal.save(args.output)


# ──────────────────────────── CLI: predict ────────────────────────────


def _write_distance_csv(
    out_path: Path,
    sample_ids: np.ndarray,
    score_col_name: str,
    scores: np.ndarray,
    threshold: float,
    serotypes: np.ndarray | None = None,
    is_capsule: np.ndarray | None = None,
) -> None:
    cols = {"sample_id": sample_ids}
    if serotypes is not None:
        cols["Serotype"] = serotypes
    if is_capsule is not None:
        cols["Is_capsule"] = is_capsule.astype(int)
    cols[score_col_name] = scores
    cols["is_novel"] = scores > threshold
    pd.DataFrame(cols).to_csv(out_path, index=False)


def _write_ranked_topk_csv(out_path: Path, sample_ids: np.ndarray, ranked: list[list[dict]]) -> None:
    rows = []
    for sid, top in zip(sample_ids, ranked):
        for r, entry in enumerate(top, start=1):
            rows.append({
                "sample_id": sid,
                "rank": r,
                "predicted_class": entry["class"],
                "mahalanobis_dist": entry["mahalanobis_dist"],
                "log_likelihood": entry["log_likelihood"],
                "softmax_prob_topk": entry["softmax_prob_topk"],
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def cli_predict(args: argparse.Namespace) -> None:
    mahal = MahalanobisOOD.load(args.mahalanobis_params)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Thresholds at the requested percentile of training scores
    threshold_mahal = float(np.percentile(mahal.train_mahalanobis_scores, args.threshold_percentile))
    threshold_rel = float(np.percentile(mahal.train_relative_mahalanobis_scores, args.threshold_percentile))

    if args.input_type == "id":
        if not args.labels:
            raise ValueError("--labels required when --input_type id")
        X, labels_df, _ = _load_id_embeddings(args.embeddings, args.labels, sep=args.sep)
        sample_ids = get_sample_id(labels_df).to_numpy()
        serotypes = labels_df["Serotype"].to_numpy()
        is_capsule = labels_df["Is_capsule"].astype(bool).to_numpy()
        topk = args.topk if args.also_ranked_id else 0
    else:  # query
        X, sample_ids = _load_query_embeddings(args.embeddings)
        serotypes = None
        is_capsule = None
        topk = args.topk

    out = mahal.score(X.astype(np.float32), topk=topk)
    _write_distance_csv(
        output_dir / f"mahalanobis_{args.input_type}_distances.csv",
        sample_ids, "mahalanobis_dist", out["mahalanobis_dist"], threshold_mahal,
        serotypes, is_capsule,
    )
    _write_distance_csv(
        output_dir / f"relative_mahalanobis_{args.input_type}_distances.csv",
        sample_ids, "relative_mahalanobis_dist", out["relative_mahalanobis_dist"], threshold_rel,
        serotypes, is_capsule,
    )
    flagged_mahal = float((out["mahalanobis_dist"] > threshold_mahal).mean())
    flagged_rel = float((out["relative_mahalanobis_dist"] > threshold_rel).mean())
    logger.info(
        "Wrote %d %s scores. M_min threshold=%.3f (p%g) → %.3f novel; "
        "RM threshold=%.3f (p%g) → %.3f novel.",
        len(X), args.input_type, threshold_mahal, args.threshold_percentile, flagged_mahal,
        threshold_rel, args.threshold_percentile, flagged_rel,
    )

    if topk > 0 and out["ranked"]:
        suffix = "id" if args.input_type == "id" else "query"
        _write_ranked_topk_csv(
            output_dir / f"mahalanobis_topk_ranked_likelihoods_{suffix}.csv",
            sample_ids, out["ranked"],
        )
        logger.info("Wrote top-%d ranked likelihoods to %s",
                    args.topk, output_dir / f"mahalanobis_topk_ranked_likelihoods_{suffix}.csv")


# ──────────────────────────── CLI: parser ────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command")

    p_fit = subparsers.add_parser("fit", help="Fit per-class Gaussians + background from ID embeddings")
    p_fit.add_argument("--embeddings", required=True,
                       help="ID .npz (dict-style, cbl|Public_ID#Contig_ID keys)")
    p_fit.add_argument("--labels", required=True, help="final_metadata.csv (CBL rows used)")
    p_fit.add_argument("--output", required=True, help="Output .pkl for fitted params")
    p_fit.add_argument("--shrinkage", default="lw", choices=["lw", "none"],
                       help="Covariance shrinkage; 'lw' = Ledoit-Wolf (recommended)")
    p_fit.add_argument("--sep", default=DEFAULT_SEP)

    p_pred = subparsers.add_parser("predict", help="Score embeddings against a fitted MahalanobisOOD")
    p_pred.add_argument("--input_type", required=True, choices=["id", "query"],
                        help="id: dict-style npz with labels; query: record_ids/embeddings npz")
    p_pred.add_argument("--embeddings", required=True)
    p_pred.add_argument("--labels", default=None,
                        help="Required when --input_type id")
    p_pred.add_argument("--mahalanobis_params", required=True, help="Path to fitted .pkl")
    p_pred.add_argument("--output_dir", required=True,
                        help="Directory where CSV outputs are written")
    p_pred.add_argument("--threshold_percentile", type=float, default=99.0,
                        help="Percentile of training scores used as the novelty threshold")
    p_pred.add_argument("--topk", type=int, default=DEFAULT_TOPK,
                        help="Number of top classes to report in ranked-likelihood output")
    p_pred.add_argument("--also_ranked_id", action="store_true",
                        help="Also write the ranked top-K likelihoods for ID samples "
                             "(off by default — large file)")
    p_pred.add_argument("--sep", default=DEFAULT_SEP)

    args = parser.parse_args()
    if args.command == "fit":
        cli_fit(args)
    elif args.command == "predict":
        cli_predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
