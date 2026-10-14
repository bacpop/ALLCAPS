"""``ALLCAPS knn`` — per-serotype novelty threshold tuning (implemented in R).

Two R scripts, run back to back:

1. ``knn_create_dataset.R`` collects the leave-one-serotype-out kNN sweep CSVs
   (``knn_query_distances_kgrid.csv`` and ``knn_id_distances_kgrid.csv``, one pair per
   held-out serotype), joins them to a ground-truth table and writes one
   ``k{K}_knn_data.csv`` per value of k.
2. ``knn_calculate_novel.R`` fits per-serotype and global ROC curves on those distances,
   picks Youden-optimal thresholds, scores them on a held-out 10%, and plots the
   distance quartiles.

This is an analysis of whether a *per-serotype* threshold beats the single global one
that ALLCAPS actually deploys. It is not part of serotyping, and it is the only part of
ALLCAPS that needs R — see environment-r.yml.
"""

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from ..logging_config import get_logger

logger = get_logger(__name__)

CREATE_DATASET_SCRIPT = "knn_create_dataset.R"
CALCULATE_NOVEL_SCRIPT = "knn_calculate_novel.R"

_NO_RSCRIPT = """Rscript was not found on your PATH.

`ALLCAPS knn` is the only command that needs R. Everything else — predict, train —
runs without it. Install the optional R environment:

    mamba env create -f environment-r.yml
    conda activate allcaps-r

then re-run this command. If R is installed somewhere unusual, pass --rscript
/path/to/Rscript."""


def _script_path(name: str) -> Path:
    """Locate a packaged R script."""
    try:
        from importlib.resources import files

        path = Path(str(files("allcaps").joinpath(name)))
    except (ImportError, ModuleNotFoundError):
        path = Path(__file__).resolve().parent.parent / name
    if not path.is_file():
        raise FileNotFoundError(f"Packaged R script missing: {path}")
    return path


def _run_r(rscript: str, script: Path, args: List[str]) -> None:
    cmd = [rscript, str(script), *args]
    logger.info("Running: %s", " ".join(cmd))
    # Streamed, not captured: these scripts are slow and chatty, and watching ggplot
    # and pROC progress is the only feedback there is.
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"{script.name} exited with status {result.returncode}. The R output above "
            f"has the detail; a missing package is the usual cause — check that the "
            f"environment-r.yml env is active."
        )


def run(
    input_dir: Path,
    ground_truth: Path,
    output: Path,
    rscript: Optional[str] = None,
    skip_dataset: bool = False,
):
    """Run both stages of the threshold-tuning analysis."""
    rscript = rscript or shutil.which("Rscript")
    if not rscript:
        raise RuntimeError(_NO_RSCRIPT)

    input_dir = Path(input_dir).expanduser()
    ground_truth = Path(ground_truth).expanduser()
    out_dir = Path(output).expanduser()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"--input must be a directory of LOO sweep CSVs: {input_dir}")
    if not ground_truth.is_file():
        raise FileNotFoundError(f"--ground-truth not found: {ground_truth}")

    sweep_csvs = list(input_dir.rglob("knn_*_distances_kgrid.csv"))
    if not sweep_csvs:
        raise FileNotFoundError(
            f"No `knn_*_distances_kgrid.csv` files under {input_dir}.\n"
            f"These come from the leave-one-serotype-out runs with a k-grid: set "
            f"`knn_k_grid` in config.yaml (e.g. 1,5,10,50) so knn_ood writes the "
            f"long-format k-grid report, then point --input at the directory holding "
            f"one subdirectory per held-out serotype."
        )
    logger.info("Found %d k-grid sweep file(s) under %s", len(sweep_csvs), input_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    if not skip_dataset:
        _run_r(
            rscript,
            _script_path(CREATE_DATASET_SCRIPT),
            [
                "--knn-raw", str(input_dir),
                "--ground-truth", str(ground_truth),
                "--output", str(out_dir),
            ],
        )

    _run_r(
        rscript,
        _script_path(CALCULATE_NOVEL_SCRIPT),
        [
            "--data-root", str(out_dir),
            "--ground-truth", str(ground_truth),
            "--output", str(out_dir),
        ],
    )

    logger.info("Threshold-tuning outputs written to %s", out_dir)
    return out_dir
