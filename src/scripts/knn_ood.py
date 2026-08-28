#!/usr/bin/env python
"""KNN-based Out-of-Distribution detection (Sun et al., ICML 2022).

OOD score = distance from the sample's feature vector to the k-th nearest
neighbor among the training (in-distribution) feature vectors. Larger distance
⇒ farther from anything we've seen ⇒ more novel. Non-parametric, model-agnostic,
and — unlike the energy score — derived from feature-space density rather than
from the closed-set classifier's logits, so it taps a different signal. This is
the deployed detector; energy is retained only as a reference baseline.

The predict step also emits, for every sample, its single closest ID neighbour
(the k=1 report): ``nn_distance`` (how far to that neighbour), ``nn_serotype``
(which trained serotype it landed next to) and ``nn_genogroup``.

Pass ``--max_k K`` to additionally write a long-format neighbour table
(``<output>_topk.csv``), one row per (sample, rank), carrying the neighbour's
index key, serotype, genogroup and distance. Useful when the top hit lands on a
serotype with too few calibration genomes to threshold reliably and you want the
runner-up. The main ``--output`` CSV is unaffected by this flag.

Pass ``--k_grid 1,5,10,50`` for the same table restricted to a handful of k
values (``<output>_kgrid.csv``, one row per (sample, k)), where the distance is
the distance to the k-th neighbour — i.e. the OOD score at that k, the quantity
``scripts.helpers.knn_k_sweep`` sweeps. Reaching k=1000 via ``--max_k`` would
write 1000 rows per sample to carry the same information; the grid writes 4.
Both reports are sliced out of a single neighbour query.

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


def _as_f64(X: np.ndarray) -> np.ndarray:
    """Promote features to float64 before any distance computation.

    Embeddings are stored as float32, but the model L2-normalises them, so
    near-duplicate loci give ``x·y ≈ 1 - 1e-8``. sklearn's cosine metric evaluates
    ``1 - x·y``, and that subtraction cancels catastrophically below float32 eps
    (~1.2e-7): ~73% of in-distribution 1-NN distances land in that band and
    quantise toward zero. Measured on the 19A fold, computing in float32 costs
    ~0.015 AUROC (0.9585 vs 0.9735); float64 recovers it exactly.
    """
    return np.ascontiguousarray(X, dtype=np.float64)


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
        X = _as_f64(X)
        n_search = min(self.k + 1, len(X))  # +1 to account for self-match
        self.index = NearestNeighbors(n_neighbors=n_search, metric=self.metric, algorithm="brute")
        self.index.fit(X)
        self.train_keys = np.asarray(keys)
        self.train_serotypes = np.asarray(serotypes) if serotypes is not None else None

        # Leave-one-out training-set k-NN distances (drop self-match at idx 0)
        dists, _ = self.index.kneighbors(X, n_neighbors=n_search, return_distance=True)
        self.train_loo_distances = dists[:, -1].astype(np.float64)
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
        X = _as_f64(X)
        n_search = self.k + 1 if query_is_id else self.k
        n_search = min(n_search, len(self.train_keys))
        dists, _ = self.index.kneighbors(X, n_neighbors=n_search, return_distance=True)
        return dists[:, -1].astype(np.float64)

    def nearest(
        self, X: np.ndarray, query_is_id: bool = False, n_neighbours: int = 1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Nearest-neighbour report: for each row, the ``n_neighbours`` closest
        training vectors — their distances, serotypes and index keys.

        Returns ``(distances, serotypes, keys)``, each shaped
        ``(len(X), n_neighbours)`` — fewer columns if the index is too small to
        supply that many — and ordered nearest-first. If ``query_is_id``
        the index was built from these same vectors, so we ask for one extra
        neighbour and drop the zero-distance self-match at column 0. When no
        serotype labels were stored at fit time the serotype array is filled
        with ``None``.

        ``n_neighbours=1`` reproduces the deployed k=1 report. ``query_is_id``
        assumes ``X`` is the same matrix, in the same row order, that the index
        was fitted on — which is how ``cli_predict`` builds it."""
        assert self.index is not None, "Call fit() first."
        assert n_neighbours >= 1, "n_neighbours must be >= 1"
        X = _as_f64(X)
        # A row has at most len(train_keys) genuine neighbours — one fewer when the
        # query IS the index, because the self-match doesn't count. Clamp rather
        # than over-request: past that point the extra column would be the
        # self-match at distance 0, landing out of order at the highest rank.
        n_available = len(self.train_keys) - (1 if query_is_id else 0)
        assert n_available >= 1, "index holds no neighbour to report"
        n_neighbours = min(n_neighbours, n_available)
        # +1 when querying the index with its own vectors, so that after dropping
        # the self-match we still have n_neighbours genuine neighbours.
        n_search = min(n_neighbours + 1 if query_is_id else n_neighbours, len(self.train_keys))
        dists, idxs = self.index.kneighbors(X, n_neighbors=n_search, return_distance=True)
        if query_is_id:
            # Exact-duplicate loci are common, so a whole block of neighbours can
            # sit at distance 0 and the self-match is NOT reliably at column 0 —
            # sklearn's ordering among ties is arbitrary. Drop it by row identity,
            # not by position. A stable argsort keeps the remaining neighbours in
            # distance order; rows where the self-match never surfaced simply lose
            # their farthest column, so every row still yields n_neighbours.
            drop = idxs == np.arange(len(X))[:, None]
            order = np.argsort(drop, axis=1, kind="stable")[:, :n_neighbours]
            dists = np.take_along_axis(dists, order, axis=1)
            idxs = np.take_along_axis(idxs, order, axis=1)
        else:
            dists, idxs = dists[:, :n_neighbours], idxs[:, :n_neighbours]

        nn_dist = dists.astype(np.float64)
        nn_keys = np.asarray(self.train_keys)[idxs]
        if self.train_serotypes is not None:
            nn_sero = np.asarray(self.train_serotypes)[idxs]
        else:
            nn_sero = np.full(idxs.shape, None, dtype=object)
        return nn_dist, nn_sero, nn_keys

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
        obj.index.fit(_as_f64(data["train_data"]))
        obj.train_keys = data["train_keys"]
        # Back-compat: indices pickled before the k=1 nearest-neighbour report
        # won't carry serotypes.
        obj.train_serotypes = data.get("train_serotypes")
        obj.train_loo_distances = data["train_loo_distances"]
        return obj


# ──────────────────────────── data loading helpers ────────────────────────────


def _load_id_embeddings(npz_path: str, labels_path: str, sep: str = DEFAULT_SEP) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Load dict-style .npz (cbl|Public_ID#Contig_ID → 128-vec) joined with labels.
    Returns (X, labels_df, keys)."""
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

    # Neighbour report. Column 0 is the deployed k=1 view: how far is each sample
    # from its single closest ID neighbour, and what serotype / genogroup does
    # that neighbour belong to? Ranks 2..max_k go to the separate top-k CSV, and
    # the --k_grid ranks to the k-grid CSV. One query, widened to whichever of the
    # two reaches further, then sliced per report.
    max_k = max(1, int(args.max_k))
    k_grid = list(args.k_grid) if args.k_grid else []
    nn_dist, nn_sero, nn_keys = knn.nearest(
        X, query_is_id=query_is_id, n_neighbours=max(max_k, *k_grid) if k_grid else max_k
    )
    out_df["nn_distance"] = nn_dist[:, 0]
    out_df["nn_serotype"] = nn_sero[:, 0]
    out_df["nn_genogroup"] = [
        map_serotype_to_group(str(s)) if s is not None else None for s in nn_sero[:, 0]
    ]

    # Mark "novel" using the percentile-of-ID convention used elsewhere in the
    # pipeline (matches the energy threshold approach in energy_summary.json).
    threshold = float(np.percentile(knn.train_loo_distances, args.threshold_percentile))
    out_df["is_novel_knn"] = out_df["knn_distance"] > threshold
    out_df.to_csv(args.output, index=False)
    logger.info(
        "Wrote %d KNN scores to %s (threshold=%.4f at p%g of train LOO distances; "
        "fraction flagged novel=%.3f)",
        len(out_df), args.output, threshold, args.threshold_percentile,
        float(out_df["is_novel_knn"].mean()),
    )

    sample_ids = out_df["sample_id"].to_numpy()
    if max_k > 1:
        _write_topk_csv(sample_ids, nn_dist, nn_sero, nn_keys, max_k, _topk_path(args))
    if k_grid:
        _write_k_grid_csv(sample_ids, nn_dist, nn_sero, nn_keys, k_grid, _k_grid_path(args))


def _sibling_path(output: str, suffix: str) -> str:
    """`<output stem>_<suffix>.<ext>`, beside the main CSV."""
    stem, _, ext = output.rpartition(".")
    return f"{stem}_{suffix}.{ext}" if stem else f"{output}_{suffix}.csv"


def _topk_path(args: argparse.Namespace) -> str:
    """Explicit --topk_output, else `<output stem>_topk.csv`."""
    return getattr(args, "topk_output", None) or _sibling_path(args.output, "topk")


def _k_grid_path(args: argparse.Namespace) -> str:
    """Explicit --k_grid_output, else `<output stem>_kgrid.csv`."""
    return getattr(args, "k_grid_output", None) or _sibling_path(args.output, "kgrid")


def _neighbour_long_df(
    sample_ids: np.ndarray,
    nn_dist: np.ndarray,
    nn_sero: np.ndarray,
    nn_keys: np.ndarray,
    ranks: list[int],
    *,
    rank_column: str,
    distance_column: str,
) -> pd.DataFrame:
    """Long/tidy neighbour table: one row per (sample, rank), in ``ranks`` order.

    ``ranks`` are 1-based neighbour ranks to keep — every rank for the top-k
    report, a sparse grid for the k-grid one. ``nn_sample_id`` is the index key
    of the training locus, so a hit can be traced back to the genome it matched.
    """
    columns = np.asarray(ranks, dtype=int) - 1
    df = pd.DataFrame({
        "sample_id": np.repeat(sample_ids, len(ranks)),
        rank_column: np.tile(ranks, len(sample_ids)),
        "nn_sample_id": nn_keys[:, columns].reshape(-1),
        "nn_serotype": nn_sero[:, columns].reshape(-1),
        distance_column: nn_dist[:, columns].reshape(-1),
    })
    # Map over the ~100 distinct serotypes, not the (n_samples x n_ranks) rows —
    # this table runs to ~500k rows for an ID sweep, where the per-row call dominates.
    groups = {s: map_serotype_to_group(str(s)) for s in pd.unique(df["nn_serotype"].dropna())}
    df["nn_genogroup"] = df["nn_serotype"].map(groups)
    return df[["sample_id", rank_column, "nn_sample_id", "nn_serotype",
               "nn_genogroup", distance_column]]


def _write_topk_csv(
    sample_ids: np.ndarray,
    nn_dist: np.ndarray,
    nn_sero: np.ndarray,
    nn_keys: np.ndarray,
    max_k: int,
    path: str,
) -> None:
    """Every rank 1..max_k. Kept separate from the deployed output so that file's
    shape never changes."""
    # A small index can hold fewer neighbours than requested; nearest() returns
    # what it has, so read the width off the array rather than trusting max_k.
    max_k = min(max_k, nn_dist.shape[1])
    topk = _neighbour_long_df(sample_ids, nn_dist, nn_sero, nn_keys,
                              list(range(1, max_k + 1)),
                              rank_column="rank", distance_column="nn_distance")
    topk.to_csv(path, index=False)
    logger.info("Wrote top-%d neighbour report (%d rows) to %s", max_k, len(topk), path)


def _write_k_grid_csv(
    sample_ids: np.ndarray,
    nn_dist: np.ndarray,
    nn_sero: np.ndarray,
    nn_keys: np.ndarray,
    k_grid: list[int],
    path: str,
) -> None:
    """Same shape as the top-k report, but only the requested k values.

    ``knn_distance`` is the distance to the k-th neighbour — the OOD score at
    that k, named to match the main CSV's column — so the file doubles as a
    per-sample k sweep without materialising every intermediate rank."""
    available = nn_dist.shape[1]
    usable = [k for k in k_grid if k <= available]
    if len(usable) < len(k_grid):
        logger.warning("k_grid values %s exceed the %d neighbours in the index — dropped",
                       [k for k in k_grid if k > available], available)
    if not usable:
        logger.warning("No usable --k_grid values; %s not written", path)
        return
    grid = _neighbour_long_df(sample_ids, nn_dist, nn_sero, nn_keys, usable,
                              rank_column="k", distance_column="knn_distance")
    grid.to_csv(path, index=False)
    logger.info("Wrote k-grid neighbour report for k=%s (%d rows) to %s",
                usable, len(grid), path)


# ──────────────────────────── CLI: parser ────────────────────────────


def _parse_k_grid(s: str) -> list[int]:
    """'1,5,10,50' → [1, 5, 10, 50], sorted and deduplicated (mirrors knn_k_sweep)."""
    grid = sorted({int(x) for x in s.split(",") if x.strip()})
    if not grid or grid[0] < 1:
        raise argparse.ArgumentTypeError(
            f"--k_grid must be a comma-separated list of positive integers, got {s!r}")
    return grid


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
    p_pred.add_argument("--max_k", type=int, default=1,
                        help="Report this many nearest neighbours. >1 writes an extra "
                             "long-format CSV (one row per sample x rank); the main "
                             "--output CSV is unchanged either way (default: 1)")
    p_pred.add_argument("--topk_output", default=None,
                        help="Path for the --max_k report (default: '<output>_topk.csv')")
    p_pred.add_argument("--k_grid", type=_parse_k_grid, default=None,
                        help="Comma-separated k values (e.g. '1,5,10,50'). Writes the same "
                             "long-format report as --max_k but with one row per (sample, k) "
                             "for these k only, where the distance is the distance to the "
                             "k-th neighbour. Off by default; the main --output CSV is "
                             "unchanged either way")
    p_pred.add_argument("--k_grid_output", default=None,
                        help="Path for the --k_grid report (default: '<output>_kgrid.csv')")
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
