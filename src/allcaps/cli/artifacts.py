"""Resolving the three files a prediction needs.

``ALLCAPS predict`` needs a trained checkpoint, a fitted kNN index and the energy
percentile table. None of them live in this repository — the checkpoint and index are too
large and are released separately — so by default they are fetched from the Hugging Face
Hub once and cached. Explicit flags always win, which is also the route for a model you
trained yourself with ``ALLCAPS train``.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..logging_config import get_logger

logger = get_logger(__name__)

#: Hub repo holding the released ALLCAPS artifacts.
DEFAULT_HF_REPO = "bacpop/ALLCAPS"

#: Filenames inside that repo. These must match the names `ALLCAPS train` writes, so a
#: locally trained run directory and a Hub snapshot are interchangeable.
MODEL_FILENAME = "transformer_model.pth"
KNN_INDEX_FILENAME = "knn_index.npz"
ENERGY_SUMMARY_FILENAME = "energy_summary.json"

#: Name of the flanking-gene FASTA shipped inside the package.
FLANKS_FILENAME = "dexB_aliA_ATCC700669.fasta"


@dataclass
class Artifacts:
    """Local paths to the files a prediction run needs."""

    model: Path
    knn_index: Optional[Path]
    energy_summary: Optional[Path]

    @property
    def can_score_knn(self) -> bool:
        return self.knn_index is not None


def packaged_flanks() -> Path:
    """Path to the bundled `dexB`/`aliA` FASTA used to cut the locus.

    Resolved from package data rather than a path relative to the working directory, so
    `--extract align` works from anywhere once ALLCAPS is installed.
    """
    try:
        from importlib.resources import files  # py3.9+

        path = Path(str(files("allcaps.data").joinpath(FLANKS_FILENAME)))
    except (ImportError, ModuleNotFoundError):
        path = Path(__file__).resolve().parent.parent / "data" / FLANKS_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"The packaged flanking-gene FASTA is missing ({path}). Reinstall ALLCAPS, "
            f"or pass --flanks with your own dexB/aliA FASTA."
        )
    return path


def cache_dir() -> Path:
    """Where downloaded artifacts land. ``ALLCAPS_CACHE`` overrides."""
    env = os.environ.get("ALLCAPS_CACHE")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "allcaps"


def _download(filename: str, repo: str, revision: Optional[str]) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is needed to download the released model. Either install "
            "it (pip install huggingface_hub) or pass --model / --knn-index / "
            "--energy-summary explicitly."
        ) from exc

    target = cache_dir()
    target.mkdir(parents=True, exist_ok=True)
    logger.info("Fetching %s from %s ...", filename, repo)
    return Path(
        hf_hub_download(
            repo_id=repo, filename=filename, revision=revision, cache_dir=str(target)
        )
    )


def _hub_hint(filename: str, repo: str, flag: str) -> str:
    return (
        f"Could not fetch {filename} from the Hugging Face repo {repo!r}.\n"
        f"\n"
        f"If you have the file locally, point at it directly:\n"
        f"    {flag} /path/to/{filename}\n"
        f"\n"
        f"`ALLCAPS train` writes all three artifacts into its output directory, so a "
        f"model you trained yourself can be used with the flags above."
    )


def resolve_artifacts(
    model: Optional[str] = None,
    knn_index: Optional[str] = None,
    energy_summary: Optional[str] = None,
    repo: str = DEFAULT_HF_REPO,
    revision: Optional[str] = None,
    offline: bool = False,
    require_knn: bool = True,
) -> Artifacts:
    """Resolve the three prediction artifacts to local paths.

    Explicit paths are used as given and must exist. Anything left unset is downloaded
    from ``repo`` unless ``offline``, in which case the missing flag is named in the
    error rather than silently falling back.
    """

    def local(path: str, what: str) -> Path:
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"{what} not found: {p}")
        return p

    # ── checkpoint: always required ──
    if model:
        model_path = local(model, "Model checkpoint")
    elif offline:
        raise FileNotFoundError(
            "--offline was given but no --model. " + _hub_hint(MODEL_FILENAME, repo, "--model")
        )
    else:
        try:
            model_path = _download(MODEL_FILENAME, repo, revision)
        except Exception as exc:
            raise FileNotFoundError(
                _hub_hint(MODEL_FILENAME, repo, "--model") + f"\n\nUnderlying error: {exc}"
            ) from exc

    # ── kNN index: needed for the deployed novelty call, but a run without it is
    #    still useful (serotype calls stand on their own), so degrade with a warning.
    knn_path: Optional[Path] = None
    if knn_index:
        knn_path = local(knn_index, "kNN index")
    elif not offline:
        try:
            knn_path = _download(KNN_INDEX_FILENAME, repo, revision)
        except Exception as exc:
            logger.warning(
                "Could not fetch %s (%s). Continuing without the kNN novelty call; "
                "is_novel_knn will be empty. Pass --knn-index to enable it.",
                KNN_INDEX_FILENAME, exc,
            )
    if knn_path is None and require_knn:
        logger.warning(
            "No kNN index available — novelty detection is DISABLED for this run. "
            "The deployed novelty call needs one; pass --knn-index, or run "
            "`ALLCAPS train` to build one."
        )

    # ── energy summary: optional. Without it the query falls back to percentiles
    #    hard-coded from a *different* model, so say so loudly rather than quietly.
    energy_path: Optional[Path] = None
    if energy_summary:
        energy_path = local(energy_summary, "Energy summary")
    elif not offline:
        try:
            energy_path = _download(ENERGY_SUMMARY_FILENAME, repo, revision)
        except Exception:
            logger.warning(
                "Could not fetch %s; the energy novelty baseline will use hard-coded "
                "percentiles from a different model and `is_novel_energy` will not be "
                "meaningful. The deployed kNN call is unaffected.",
                ENERGY_SUMMARY_FILENAME,
            )

    return Artifacts(model=model_path, knn_index=knn_path, energy_summary=energy_path)
