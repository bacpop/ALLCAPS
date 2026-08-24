"""Regenerate the per-serotype threshold-accuracy plots from a finished k-sweep,
without re-running the (expensive, cluster-only) sweep itself.

Reads ``k_sweep_best_per_fold.csv`` produced by ``scripts.helpers.knn_k_sweep``
and redraws both figures:
  - threshold_accuracy_per_serotype        (ceiling: per-serotype optimal k & threshold)
  - threshold_accuracy_per_serotype_deploy (deployment: fixed k, p99-of-ID threshold)
Use this to iterate on the figures locally. The balanced-accuracy / deployment
columns must already exist in the CSV (added by a k-sweep run that post-dates
this feature).

Usage:
  python -m scripts.helpers.plot_knn_threshold_accuracy \\
      --sweep_dir results/knn-sweep
"""

import argparse
from pathlib import Path

import pandas as pd

from .knn_k_sweep import _plot_ceiling_accuracy, _plot_deploy_accuracy
from ..logging_config import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sweep_dir", required=True,
                   help="Directory holding k_sweep_best_per_fold.csv (sweep output)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    best_csv = sweep_dir / "k_sweep_best_per_fold.csv"
    if not best_csv.exists():
        raise FileNotFoundError(
            f"{best_csv} not found — point --sweep_dir at a knn_k_sweep output dir."
        )
    best = pd.read_csv(best_csv)

    _plot_ceiling_accuracy(best, sweep_dir / "threshold_accuracy_per_serotype")
    # Recover the deployment k / percentile recorded by the sweep (fall back to defaults).
    deploy_k = 1  # int(best["deploy_k"].dropna().iloc[0]) if "deploy_k" in best else 5
    pct = float(best["deploy_threshold_percentile"].dropna().iloc[0]) if "deploy_threshold_percentile" in best else 99.0
    _plot_deploy_accuracy(best, sweep_dir / "threshold_accuracy_per_serotype_deploy", deploy_k, pct)
    logger.info("Wrote threshold_accuracy_per_serotype{,_deploy}.{png,pdf} (%d serotypes; deploy k=%d, p%g)",
                len(best), deploy_k, pct)


if __name__ == "__main__":
    main()
