"""The ``ALLCAPS`` command.

Three subcommands:

* ``predict`` — assemblies in, serotype and novelty calls out. The common case.
* ``train``   — assemblies plus serotypes in, a usable model out.
* ``knn``     — per-serotype novelty threshold tuning (needs R).
"""

import json
from pathlib import Path
from typing import List, Optional

import typer

from ..consts import DEFAULT_MODEL
from ..logging_config import get_logger
from . import artifacts as art
from . import manifest as mf
from .train import RELEASED_TRAIN_DEFAULTS

logger = get_logger(__name__)

app = typer.Typer(
    name="ALLCAPS",
    help=(
        "Serotype pneumococcal capsular loci from genome assemblies, and flag loci "
        "whose serotype is novel.\n\n"
        "Quick start:\n\n"
        "    ALLCAPS predict --input samples.txt --extract align --output results/\n\n"
        "where samples.txt lists one assembly FASTA path per line."
    ),
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _version_callback(value: bool):
    if value:
        try:
            from importlib.metadata import version

            typer.echo(f"ALLCAPS {version('allcaps')}")
        except Exception:
            typer.echo("ALLCAPS (version unknown — not installed as a package)")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True,
        help="Show the version and exit.",
    ),
):
    pass


def _collect(input_: Optional[Path], paths: Optional[List[str]],
             require_serotype: bool) -> List[mf.Sample]:
    """Resolve --input / bare paths into samples, or fail with guidance."""
    if input_ and paths:
        raise typer.BadParameter(
            "Give either --input (a manifest) or bare FASTA paths, not both."
        )
    if input_:
        return mf.read_manifest(input_, require_serotype=require_serotype)
    if paths:
        return mf.from_paths(paths)
    raise typer.BadParameter(
        "No input. Pass --input with a manifest file, or list FASTA paths directly:\n"
        "    ALLCAPS predict --input samples.txt --extract align\n"
        "    ALLCAPS predict a.fasta b.fasta --extract align"
    )


def _fail(exc: Exception) -> None:
    """Report a failure as a message, not a traceback."""
    typer.secho(f"\nError: {exc}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


# ══════════════════════════════════════════════════════════════
#  predict
# ══════════════════════════════════════════════════════════════


@app.command(
    help=(
        "Serotype one or more assemblies.\n\n"
        "--extract chooses how the cps locus is found and must be given explicitly:\n\n"
        "  align  cut between the dexB/aliA flanking genes with minimap2. Fast, and "
        "what the model was trained on. Use this.\n\n"
        "  scan   embed rolling windows across the whole assembly, no flanks needed. "
        "Use only when the flanks are genuinely absent — it is orders of magnitude "
        "slower (hours per genome on CPU)."
    )
)
def predict(
    paths: Optional[List[str]] = typer.Argument(
        None, metavar="[FASTA]...", help="Assembly FASTA paths (alternative to --input)."
    ),
    input: Optional[Path] = typer.Option(
        None, "--input", "-i",
        help="Manifest: one FASTA path per line, or a CSV/TSV with a `path` column.",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Directory for predictions.tsv. Without it, results go to stdout only.",
    ),
    extract: str = typer.Option(
        ..., "--extract", "-e",
        help="How to locate the cps locus: 'align' (recommended) or 'scan'.",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Checkpoint path. Default: download the released model."
    ),
    knn_index: Optional[str] = typer.Option(
        None, "--knn-index", help="Fitted kNN index. Default: download the released one."
    ),
    energy_summary: Optional[str] = typer.Option(
        None, "--energy-summary", help="energy_summary.json for the energy baseline."
    ),
    flanks: Optional[str] = typer.Option(
        None, "--flanks",
        help="FASTA of flanking genes for --extract align. Default: the packaged dexB/aliA.",
    ),
    device: str = typer.Option("auto", "--device", help="cuda, cpu, or auto."),
    base_model: str = typer.Option(DEFAULT_MODEL, "--base-model", help="ProkBERT model id."),
    cutoff: float = typer.Option(
        0.7, "--cutoff", help="Min alignment length fraction for a flank hit (align)."
    ),
    max_extension: int = typer.Option(
        30_000, "--max-extension",
        help="Max bases to extend toward a contig end when only one flank hits (align).",
    ),
    scan_step: int = typer.Option(
        2000, "--scan-step", help="Rolling-window step in bp (scan)."
    ),
    threshold_percentile: float = typer.Option(
        95.0, "--threshold-percentile",
        help="Novelty threshold as a percentile of training leave-one-out 1-NN distances.",
    ),
    energy_percentile: float = typer.Option(
        93.0, "--energy-percentile",
        help="Operating point for the energy baseline, as a percentile key of "
        "energy_summary.json (the released file carries 93, 95, 99 and 99.5). Affects "
        "is_novel_energy only — the deployed kNN novelty call is unaffected.",
    ),
    max_k: int = typer.Option(5, "--max-k", help="Neighbours to report per locus."),
    hf_repo: str = typer.Option(art.DEFAULT_HF_REPO, "--hf-repo", help="Hub repo for artifacts."),
    revision: Optional[str] = typer.Option(None, "--revision", help="Hub revision/tag."),
    offline: bool = typer.Option(False, "--offline", help="Never download; require explicit paths."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Do not print the table to stdout."),
    keep_intermediates: bool = typer.Option(
        False, "--keep-intermediates", help="Keep cut loci, embeddings and raw CSVs."
    ),
):
    from . import predict as predict_impl

    if extract not in ("align", "scan"):
        raise typer.BadParameter("--extract must be 'align' or 'scan'.", param_hint="--extract")
    try:
        samples = _collect(input, paths, require_serotype=False)
        predict_impl.run(
            samples=samples, extract=extract, output=output, model=model,
            knn_index=knn_index, energy_summary=energy_summary, flanks=flanks,
            device=device, base_model=base_model, cutoff=cutoff,
            max_extension=max_extension, scan_step=scan_step,
            threshold_percentile=threshold_percentile,
            energy_percentile=energy_percentile, max_k=max_k, hf_repo=hf_repo,
            revision=revision, offline=offline, quiet=quiet,
            keep_intermediates=keep_intermediates,
        )
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(exc)


# ══════════════════════════════════════════════════════════════
#  train
# ══════════════════════════════════════════════════════════════


@app.command(
    help=(
        "Train a model from assemblies with known serotypes.\n\n"
        "--input is a CSV/TSV with a `path` column and a `serotype` column, one row per "
        "assembly. The cps locus is cut from each assembly with the dexB/aliA flanks, "
        "and the leftover fragments become the non-capsular class, so no is_cbl column "
        "is needed.\n\n"
        "Every default reproduces the released checkpoint, so the bare command retrains "
        "the published recipe on your data.\n\n"
        "This is a long job — the released checkpoint took about 19 hours on one A100. "
        "The output directory ends up holding the checkpoint, the kNN index and "
        "energy_summary.json, which is everything `predict` needs."
    )
)
def train(
    input: Path = typer.Option(
        ..., "--input", "-i",
        help="CSV/TSV manifest with `path` and `serotype` columns.",
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Run directory."),
    flanks: Optional[str] = typer.Option(None, "--flanks", help="Flanking-gene FASTA."),
    device: str = typer.Option("auto", "--device", help="cuda, cpu, or auto."),
    base_model: str = typer.Option(DEFAULT_MODEL, "--base-model", help="ProkBERT model id."),
    epochs: int = typer.Option(RELEASED_TRAIN_DEFAULTS["epochs"], "--epochs"),
    batch_size: int = typer.Option(RELEASED_TRAIN_DEFAULTS["batch_size"], "--batch-size"),
    lr: float = typer.Option(RELEASED_TRAIN_DEFAULTS["lr"], "--lr"),
    model_params: Optional[str] = typer.Option(
        None, "--model-params",
        help="JSON object of hyperparameters, merged over the released defaults.",
    ),
    aug_noise_std: float = typer.Option(
        RELEASED_TRAIN_DEFAULTS["aug_noise_std"], "--aug-noise-std",
        help="Gaussian noise std added to chunk embeddings while training. Released "
        "value; 0 disables.",
    ),
    aug_chunk_dropout: float = typer.Option(
        RELEASED_TRAIN_DEFAULTS["aug_chunk_dropout"], "--aug-chunk-dropout",
        help="Probability of dropping a chunk while training. Released value; 0 disables.",
    ),
    aug_spec_freq: float = typer.Option(
        RELEASED_TRAIN_DEFAULTS["aug_spec_freq"], "--aug-spec-freq",
        help="SpecAugment frequency-masking probability. Released value; 0 disables. "
        "Setting this, --aug-noise-std and --aug-chunk-dropout all to 0 turns "
        "augmentation off entirely.",
    ),
    aug_spec_width: int = typer.Option(
        RELEASED_TRAIN_DEFAULTS["aug_spec_width"], "--aug-spec-width",
        help="SpecAugment maximum mask width, in feature dimensions.",
    ),
    aug_n_views: int = typer.Option(
        RELEASED_TRAIN_DEFAULTS["aug_n_views"], "--aug-n-views",
        help="Augmented views per sample (1 = the original only).",
    ),
    split_ratio: float = typer.Option(
        0.9, "--split-ratio", help="Train fraction, counted in samples not contigs."
    ),
    cutoff: float = typer.Option(0.7, "--cutoff", help="Flank alignment length cutoff."),
    max_extension: int = typer.Option(30_000, "--max-extension"),
    threshold_percentile: float = typer.Option(95.0, "--threshold-percentile"),
    knn_k: int = typer.Option(1, "--knn-k", help="k for the novelty index."),
    seq_max_len: int = typer.Option(
        30_000, "--seq-max-len",
        help="Truncation window per contig, in bp. Changing this changes what the model "
        "learns — the released checkpoint used 30000.",
    ),
    records_per_batch: int = typer.Option(
        32, "--records-per-batch", help="Records buffered before a ProkBERT flush."
    ),
    max_chunks_per_batch: int = typer.Option(
        64, "--max-chunks-per-batch", help="Chunk batch cap; lower it if you hit OOM."
    ),
    wandb: bool = typer.Option(
        False, "--wandb",
        help="Upload metrics to Weights & Biases. Off by default; needs allcaps[wandb].",
    ),
    resume: bool = typer.Option(
        False, "--resume", help="Skip stages whose outputs already exist."
    ),
    seed: int = typer.Option(42, "--seed"),
):
    from . import train as train_impl

    try:
        params = json.loads(model_params) if model_params else None
        if params is not None and not isinstance(params, dict):
            raise typer.BadParameter("--model-params must be a JSON object.")
        samples = mf.read_manifest(input, require_serotype=True)
        train_impl.run(
            samples=samples, output=output, flanks=flanks, device=device,
            base_model=base_model, epochs=epochs, batch_size=batch_size, lr=lr,
            aug_noise_std=aug_noise_std, aug_chunk_dropout=aug_chunk_dropout,
            aug_spec_freq=aug_spec_freq, aug_spec_width=aug_spec_width,
            aug_n_views=aug_n_views,
            model_params=params, split_ratio=split_ratio, cutoff=cutoff,
            max_extension=max_extension, threshold_percentile=threshold_percentile,
            knn_k=knn_k, seq_max_len=seq_max_len, records_per_batch=records_per_batch,
            max_chunks_per_batch=max_chunks_per_batch, wandb=wandb, resume=resume,
            seed=seed,
        )
    except typer.Exit:
        raise
    except json.JSONDecodeError as exc:
        _fail(ValueError(f"--model-params is not valid JSON: {exc}"))
    except Exception as exc:
        _fail(exc)


# ══════════════════════════════════════════════════════════════
#  knn
# ══════════════════════════════════════════════════════════════


@app.command(
    help=(
        "Tune per-serotype novelty thresholds from leave-one-serotype-out sweeps.\n\n"
        "An analysis command, not part of serotyping: it asks whether a per-serotype "
        "threshold would beat the single global one ALLCAPS deploys. Implemented in R, "
        "and the only command that needs it — see environment-r.yml."
    )
)
def knn(
    input: Path = typer.Option(
        ..., "--input", "-i",
        help="Directory of LOO sweep CSVs, one subdirectory per held-out serotype.",
    ),
    ground_truth: Path = typer.Option(
        ..., "--ground-truth", "-g",
        help="CSV of sample_id, Serotype, Serogroup, dataset, benchmark.",
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory."),
    rscript: Optional[str] = typer.Option(
        None, "--rscript", help="Path to Rscript. Default: whatever is on PATH."
    ),
    skip_dataset: bool = typer.Option(
        False, "--skip-dataset",
        help="Reuse existing k*_knn_data.csv and only re-run the tuning stage.",
    ),
):
    from . import knn as knn_impl

    try:
        knn_impl.run(
            input_dir=input, ground_truth=ground_truth, output=output,
            rscript=rscript, skip_dataset=skip_dataset,
        )
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(exc)


if __name__ == "__main__":
    app()
