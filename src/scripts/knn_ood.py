#!/usr/bin/env python
"""KNN-based Out-of-Distribution detection (Sun et al., ICML 2022).

OOD score = distance from the sample's feature vector to the k-th nearest
neighbor among the training (in-distribution) feature vectors. Larger distance
⇒ farther from anything we've seen ⇒ more novel. Non-parametric, model-agnostic,
and — unlike energy and OpenMax — derived from feature-space density rather
than from the closed-set classifier's logits / MAVs, so it taps a different
signal from the methods already deployed.

The predict step also emits, for every sample, its single closest ID neighbour
(the k=1 report): ``nn_distance`` (how far to that neighbour), ``nn_serotype``
(which trained serotype it landed next to) and ``nn_genogroup``.

Usage:
  # Fit (build index from ID training embeddings):
  python -m scripts.knn_ood fit \\
      --embeddings results/inference_results.npz \\
      --labels results/final_metadata.csv \\
      --output results/knn_index.pkl \\
      --k 1

  # Predict on ID embeddings (for cross-method AUROC):
  python -m scripts.knn_ood predict \\
      --input_type id \\
      --embeddings results/inference_results.npz \\
      --labels results/final_metadata.csv \\
      --knn_index results/knn_index.pkl \\
      --output results/knn_id_distances.csv

  # Predict on query (held-out novel) embeddings:
  python -m scripts.knn_ood predict \\
      --input_type query \\
      --embeddings results/test_output/query_embeddings.npz \\
      --knn_index results/knn_index.pkl \\
      --output results/knn_query_distances.csv
"""

import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .consts import DEFAULT_MISSING_LABEL, DEFAULT_SEP
from .logging_config import get_logger
from .utils import get_sample_id, map_serotype_to_group

logger = get_logger(__name__)

DEFAULT_K = 50
DEFAULT_METRIC = "cosine"


# ──────────────────────────── core ────────────────────────────


class KnnOOD:
    """Wraps sklearn NearestNeighbors with a fit-time normalization and a
    distance-to-k-th-NN scoring rule."""

    def __init__(self, k: int = DEFAULT_K, metric: str = DEFAULT_METRIC):
        self.k = k
        self.metric = metric
        self.index: NearestNeighbors | None = None
        self.train_keys: np.ndarray | None = None
        # Serotype label of every training (ID) vector, aligned row-for-row with
        # the index. Lets us report *which* ID serotype a query lands next to
        # (the k=1 "nearest neighbour" diagnostic).
        self.train_serotypes: np.ndarray | None = None
        # For diagnostics: the empirical k-th NN distance distribution on
        # the training set itself (leave-one-out). Used by callers to pick
        # a percentile-based threshold consistent with the energy convention.
        self.train_loo_distances: np.ndarray | None = None

    def fit(self, X: np.ndarray, keys: np.ndarray, serotypes: np.ndarray | None = None) -> None:
        assert X.ndim == 2 and X.shape[0] == len(keys), "X / keys mismatch"
        assert serotypes is None or len(serotypes) == len(keys), "serotypes / keys mismatch"
        n_search = min(self.k + 1, len(X))  # +1 to account for self-match
        self.index = NearestNeighbors(n_neighbors=n_search, metric=self.metric, algorithm="brute")
        self.index.fit(X)
        self.train_keys = np.asarray(keys)
        self.train_serotypes = np.asarray(serotypes) if serotypes is not None else None

        # Leave-one-out training-set k-NN distances (drop self-match at idx 0)
        dists, _ = self.index.kneighbors(X, n_neighbors=n_search, return_distance=True)
        self.train_loo_distances = dists[:, -1].astype(np.float32)
        logger.info(
            "KNN fit: n_train=%d, k=%d, metric=%s, ID kth-NN dist median=%.4f, p99=%.4f",
            len(X), self.k, self.metric,
            float(np.median(self.train_loo_distances)),
            float(np.percentile(self.train_loo_distances, 99)),
        )

    def score(self, X: np.ndarray, query_is_id: bool = False) -> np.ndarray:
        """Distance to the k-th nearest training neighbor. Higher ⇒ more novel.

        If ``query_is_id``, the index was built from these same vectors; we
        request k+1 neighbors and discard the self-match (always idx 0,
        distance 0)."""
        assert self.index is not None, "Call fit() first."
        n_search = self.k + 1 if query_is_id else self.k
        n_search = min(n_search, len(self.train_keys))
        dists, _ = self.index.kneighbors(X, n_neighbors=n_search, return_distance=True)
        return dists[:, -1].astype(np.float32)

    def nearest(self, X: np.ndarray, query_is_id: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """k=1 nearest-neighbour report: distance to the single closest training
        vector and that neighbour's serotype.

        Returns ``(distances, serotypes)``. If ``query_is_id`` the index was
        built from these same vectors, so we ask for 2 neighbours and skip the
        zero-distance self-match at column 0. When no serotype labels were
        stored at fit time the serotype array is filled with ``None``."""
        assert self.index is not None, "Call fit() first."
        n_search = min(2 if query_is_id else 1, len(self.train_keys))
        dists, idxs = self.index.kneighbors(X, n_neighbors=n_search, return_distance=True)
        col = 1 if query_is_id else 0
        nn_dist = dists[:, col].astype(np.float32)
        nn_idx = idxs[:, col]
        if self.train_serotypes is not None:
            nn_sero = self.train_serotypes[nn_idx]
        else:
            nn_sero = np.array([None] * len(nn_idx), dtype=object)
        return nn_dist, nn_sero

    def save(self, path: str) -> None:
        assert self.index is not None, "Cannot save unfitted KnnOOD."
        with open(path, "wb") as f:
            pickle.dump({
                "k": self.k,
                "metric": self.metric,
                "train_data": self.index._fit_X,
                "train_keys": self.train_keys,
                "train_serotypes": self.train_serotypes,
                "train_loo_distances": self.train_loo_distances,
            }, f)
        logger.info("KNN index saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "KnnOOD":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(k=data["k"], metric=data["metric"])
        n_search = min(obj.k + 1, len(data["train_data"]))
        obj.index = NearestNeighbors(n_neighbors=n_search, metric=obj.metric, algorithm="brute")
        obj.index.fit(data["train_data"])
        obj.train_keys = data["train_keys"]
        # Back-compat: indices pickled before the k=1 nearest-neighbour report
        # won't carry serotypes.
        obj.train_serotypes = data.get("train_serotypes")
        obj.train_loo_distances = data["train_loo_distances"]
        return obj


# ──────────────────────────── data loading helpers ────────────────────────────


def _load_id_embeddings(npz_path: str, labels_path: str, sep: str = DEFAULT_SEP) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Load dict-style .npz (cbl|Public_ID#Contig_ID → 128-vec) joined with labels.
    Returns (X, labels_df, keys) — same convention as openmax.cli_fit."""
    X = np.load(npz_path, allow_pickle=True)
    labels_df = pd.read_csv(
        labels_path, index_col=0,
        sep="\t" if labels_path.endswith(".tsv") else ",",
    )
    labels_df["Serotype"] = labels_df["Serotype"].fillna(DEFAULT_MISSING_LABEL)
    labels_df = labels_df[labels_df["Serotype"] != DEFAULT_MISSING_LABEL]
    labels_df = labels_df[labels_df["Is_capsule"].astype(bool)]

    keys = (
        labels_df["Is_capsule"].map(lambda x: "cbl" if x else "non-cbl")
        + sep
        + get_sample_id(labels_df)
    )
    valid_keys = [k for k in keys if k in X]
    valid_mask = keys.isin(valid_keys)
    labels_df = labels_df[valid_mask.values]
    keys = keys[valid_mask.values].to_numpy()
    X_filtered = np.stack([X[k] for k in keys])
    logger.info("Loaded %d capsulated ID embeddings from %s", len(X_filtered), npz_path)
    return X_filtered, labels_df, keys


def _load_query_embeddings(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load query-style .npz with `record_ids` and `embeddings` keys."""
    Y = np.load(npz_path, allow_pickle=True)
    if "record_ids" not in Y or "embeddings" not in Y:
        raise ValueError(
            f"{npz_path} does not have expected 'record_ids'/'embeddings' arrays "
            f"(found: {list(Y.files)})"
        )
    return Y["embeddings"].astype(np.float32), np.asarray(Y["record_ids"])


# ──────────────────────────── CLI: fit ────────────────────────────


def cli_fit(args: argparse.Namespace) -> None:
    X, labels_df, keys = _load_id_embeddings(args.embeddings, args.labels, sep=args.sep)
    knn = KnnOOD(k=args.k, metric=args.distance_metric)
    knn.fit(X, keys, serotypes=labels_df["Serotype"].to_numpy())
    knn.save(args.output)


# ──────────────────────────── CLI: predict ────────────────────────────


def cli_predict(args: argparse.Namespace) -> None:
    knn = KnnOOD.load(args.knn_index)

    if args.input_type == "id":
        if not args.labels:
            raise ValueError("--labels is required when --input_type id")
        X, labels_df, keys = _load_id_embeddings(args.embeddings, args.labels, sep=args.sep)
        # Self-match expected → request k+1 and discard the zero-distance hit.
        scores = knn.score(X, query_is_id=True)
        out_df = pd.DataFrame({
            "sample_id": get_sample_id(labels_df).to_numpy(),
            "Serotype": labels_df["Serotype"].to_numpy(),
            "Is_capsule": labels_df["Is_capsule"].astype(int).to_numpy(),
            "knn_distance": scores,
        })
        query_is_id = True
    else:  # query
        X, record_ids = _load_query_embeddings(args.embeddings)
        scores = knn.score(X, query_is_id=False)
        out_df = pd.DataFrame({
            "sample_id": record_ids,
            "knn_distance": scores,
        })
        query_is_id = False

    # k=1 report: how far is each sample from its single closest ID neighbour,
    # and what serotype / genogroup does that neighbour belong to?
    nn_dist, nn_sero = knn.nearest(X, query_is_id=query_is_id)
    out_df["nn_distance"] = nn_dist
    out_df["nn_serotype"] = nn_sero
    out_df["nn_genogroup"] = [
        map_serotype_to_group(str(s)) if s is not None else None for s in nn_sero
    ]

    # Mark "novel" using the percentile-of-ID convention used elsewhere in the
    # pipeline (matches the energy threshold approach in energy_summary.json).
    threshold = float(np.percentile(knn.train_loo_distances, args.threshold_percentile))
    out_df["is_novel"] = out_df["knn_distance"] > threshold
    out_df.to_csv(args.output, index=False)
    logger.info(
        "Wrote %d KNN scores to %s (threshold=%.4f at p%g of train LOO distances; "
        "fraction flagged novel=%.3f)",
        len(out_df), args.output, threshold, args.threshold_percentile,
        float(out_df["is_novel"].mean()),
    )


# ──────────────────────────── CLI: parser ────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command")

    p_fit = subparsers.add_parser("fit", help="Build a KNN index from ID embeddings")
    p_fit.add_argument("--embeddings", required=True,
                       help="ID .npz (dict-style, cbl|Public_ID#Contig_ID keys)")
    p_fit.add_argument("--labels", required=True, help="final_metadata.csv (CBL rows used)")
    p_fit.add_argument("--output", required=True, help="Output .pkl for the index")
    p_fit.add_argument("--k", type=int, default=DEFAULT_K,
                       help="k for distance-to-kth-NN scoring (default: 50)")
    p_fit.add_argument("--distance_metric", default=DEFAULT_METRIC,
                       choices=["cosine", "euclidean"])
    p_fit.add_argument("--sep", default=DEFAULT_SEP)

    p_pred = subparsers.add_parser("predict", help="Score embeddings against a fitted index")
    p_pred.add_argument("--input_type", required=True, choices=["id", "query"],
                        help="id: dict-style npz with labels; query: record_ids/embeddings npz")
    p_pred.add_argument("--embeddings", required=True)
    p_pred.add_argument("--labels", default=None,
                        help="Required when --input_type id")
    p_pred.add_argument("--knn_index", required=True, help="Path to fitted .pkl")
    p_pred.add_argument("--output", required=True, help="Output CSV path")
    p_pred.add_argument("--threshold_percentile", type=float, default=99.0,
                        help="Percentile of training k-th-NN distances used as the novelty threshold")
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
