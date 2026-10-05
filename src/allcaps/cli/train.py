"""``ALLCAPS train`` — assemblies + serotypes in, a usable model out.

Chains the same stages the Snakefile runs, in the same order, but driven from a manifest
of raw assemblies rather than a hand-written config. The run directory ends up holding
exactly the three artifacts ``ALLCAPS predict`` consumes, so a model trained here can be
used immediately::

    transformer_model.pth    the checkpoint
    knn_index.npz            the fitted novelty index (pickle-free)
    energy_summary.json      energy percentiles for the reference baseline

This is a long job. The released checkpoint took roughly 19 hours on one A100, most of
it in the 5-fold cross-validation inside `train_model`. Use `--resume` to pick up a run
that stopped partway.
"""

import json
from pathlib import Path
from typing import List, Optional

from ..consts import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EPOCHS,
    DEFAULT_HEAD_MODEL,
    DEFAULT_LR,
    DEFAULT_MAX_LEN,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIM,
    DEFAULT_STRIDE_RATIO,
    TRAIN_SPLIT_RATIO,
)
from ..logging_config import get_logger
from ._invoke import ns, resolve_device, stage
from .manifest import Sample

logger = get_logger(__name__)

#: Hyperparameters of the released checkpoint. `alpha: 0` disables the contrastive
#: term; both are load-bearing and documented in TRAINING.md.
RELEASED_MODEL_PARAMS = {
    "embedding_dim": DEFAULT_EMBEDDING_DIM,
    "output_dim": DEFAULT_OUTPUT_DIM,
    "num_layers": 1,
    "nhead": 4,
    "k_folds": 5,
    "random_state": 42,
    "temperature": 0.07,
    "alpha": 0,
    "dataset_name": "multidomain_chunked",
}


def _write_pipeline_metadata(samples: List[Sample], data_dir: Path) -> Path:
    """Translate the manifest into the metadata CSV the label stage reads.

    ``data_labels_preprocessing.read_monocle_metadata`` selects exactly the columns
    ``ERR`` and ``In_silico_serotype``, so the manifest is rendered into that shape here
    rather than loosening a reader that both training and evaluation depend on.
    """
    import pandas as pd

    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "manifest_metadata.csv"
    pd.DataFrame(
        {
            "ERR": [s.sample_id for s in samples],
            "In_silico_serotype": [s.serotype for s in samples],
        }
    ).to_csv(path, index=False)
    logger.info("Wrote %d manifest label row(s) to %s", len(samples), path)
    return path


def run(
    samples: List[Sample],
    output: Path,
    flanks: Optional[str] = None,
    device: str = "auto",
    base_model: str = DEFAULT_MODEL,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    model_params: Optional[dict] = None,
    split_ratio: float = TRAIN_SPLIT_RATIO,
    cutoff: float = 0.7,
    max_extension: int = 30_000,
    threshold_percentile: float = 95.0,
    knn_k: int = 1,
    seq_max_len: int = DEFAULT_MAX_LEN,
    records_per_batch: int = 32,
    max_chunks_per_batch: int = 64,
    wandb: bool = False,
    resume: bool = False,
    seed: int = 42,
):
    from .. import (
        data_labels_postprocessing,
        data_labels_preprocessing,
        data_locus_cutter,
        embed_transformer,
        eval_serotype_classifier,
        knn_ood,
    )
    from ..helpers import data_train_test_split
    from ..tracking import tracking
    from ..trihead import infer_trihead_transformer, train_trihead_transformer
    from . import artifacts as art

    device = resolve_device(device)
    params = dict(RELEASED_MODEL_PARAMS)
    if model_params:
        params.update(model_params)
    if "embedding_dim" not in params:
        raise ValueError("model_params must include embedding_dim.")

    run_dir = Path(output).expanduser()
    data_dir = run_dir / "data"
    emb_dir = data_dir / "base_embeddings_chunked"
    run_dir.mkdir(parents=True, exist_ok=True)

    def maybe(paths):
        """Outputs to check for `--resume`; empty disables skipping."""
        return paths if resume else ()

    flank_path = Path(flanks).expanduser() if flanks else art.packaged_flanks()
    logger.info("Run directory: %s (device=%s)", run_dir, device)

    # ── 1. cut the cps locus out of each assembly ──
    infiles = data_dir / "infiles.txt"
    data_dir.mkdir(parents=True, exist_ok=True)
    infiles.write_text("".join(f"{s.path}\n" for s in samples))
    cbl = data_dir / "contigs.fasta"
    noncbl = data_dir / "contigs_noncbl.fasta"
    stage(
        "locus_cutting",
        data_locus_cutter.main,
        ns(
            data_locus_cutter.build_parser,
            infiles=str(infiles),
            query=str(flank_path),
            cutoff=cutoff,
            outpref=str(data_dir / "contigs"),
            max_extension=max_extension,
            save_noncbl=True,
        ),
        skip_if=maybe([cbl, noncbl]),
    )

    # ── 2. clean the labels ──
    metadata = _write_pipeline_metadata(samples, data_dir)
    cleaned = run_dir / "cleaned_labels.csv"
    stage(
        "labels_preprocessing",
        data_labels_preprocessing.main,
        ns(
            data_labels_preprocessing.build_parser,
            metadata=str(metadata),
            output_dir=str(run_dir),
            cbl_fasta=str(cbl),
            noncbl_fasta=str(noncbl),
        ),
        skip_if=maybe([cleaned]),
    )

    # ── 3. split by sample, never by contig ──
    # `initial_metadata.csv` (not cleaned_labels.csv) is the per-contig table; the
    # splitter needs Contig_ID, which cleaned_labels.csv does not carry.
    initial_metadata = run_dir / "initial_metadata.csv"
    train_fasta = data_dir / "train.fasta"
    stage(
        "train_test_split",
        data_train_test_split.main,
        ns(
            data_train_test_split.build_parser,
            fastas=[str(cbl), str(noncbl)],
            metadata=[str(initial_metadata), str(initial_metadata)],
            ratios=str(split_ratio),
            output_dir=str(data_dir),
            seed=seed,
        ),
        skip_if=maybe([train_fasta]),
    )

    # ── 4. ProkBERT chunk embeddings (slow) ──
    stage(
        "embed_base",
        embed_transformer.main,
        ns(
            embed_transformer.build_parser,
            fasta=str(train_fasta),
            out_dir=str(emb_dir),
            device=device,
            model_name=base_model,
            chunk_size=DEFAULT_CHUNK_SIZE,
            stride_ratio=DEFAULT_STRIDE_RATIO,
            seq_max_len=seq_max_len,
            records_per_batch=records_per_batch,
            max_chunks_per_batch=max_chunks_per_batch,
        ),
        # embed_transformer is itself resumable (it skips existing .npy), so it is
        # always re-entered rather than skipped wholesale.
    )

    # ── 5. drop label rows with no embedding ──
    final_metadata = data_dir / "final_metadata.csv"
    stage(
        "labels_postprocessing",
        data_labels_postprocessing.main,
        ns(
            data_labels_postprocessing.build_parser,
            clean_labels=str(data_dir / "train_metadata.csv"),
            embedding_dir=str(emb_dir),
            output_dir=str(data_dir),
            skip_labels=[],
        ),
        skip_if=maybe([final_metadata]),
    )

    # ── 6. train ──
    checkpoint = run_dir / "transformer_model.pth"
    tracking.start(config={"epochs": epochs, "lr": lr, **params}, enabled=wandb)
    try:
        stage(
            "train_model",
            train_trihead_transformer.main,
            ns(
                train_trihead_transformer.build_parser,
                embedding_dir=str(emb_dir),
                labels=str(final_metadata),
                output=str(checkpoint),
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                model_params=params,
                labeled_only=True,
                hierarchical_loss=True,
                skip_labels=[],
                wandb=wandb,
            ),
            skip_if=maybe([checkpoint]),
        )
    finally:
        tracking.finish()

    # ── 7. embed the training set through the trained model ──
    inference_npz = run_dir / "inference_results.npz"
    stage(
        "embed_chunks",
        infer_trihead_transformer.main,
        ns(
            infer_trihead_transformer.build_parser,
            embeddings_dir=str(emb_dir),
            labels=str(final_metadata),
            model=str(checkpoint),
            output=str(inference_npz),
            device=device,
            batch_size=batch_size,
            model_params=params,
            labeled_only=True,
            skip_labels=[],
        ),
        skip_if=maybe([inference_npz]),
    )

    # ── 8. closed-set performance, and the energy percentile table ──
    # `--collect_energies --save_energy_summary` is what writes energy_summary.json.
    # No Snakemake rule passes these, which is why the published query path has been
    # falling back to percentiles hard-coded from an older model.
    energy_summary = run_dir / "energy_summary.json"
    stage(
        "serotype_classification",
        eval_serotype_classifier.main,
        ns(
            eval_serotype_classifier.build_parser,
            embeddings=str(inference_npz),
            labels=str(final_metadata),
            model=str(checkpoint),
            output_dir=str(run_dir),
            device=device,
            batch_size=batch_size,
            model_params=params,
            collect_energies=True,
            save_energy_summary=True,
        ),
        skip_if=maybe([energy_summary]),
    )

    # ── 9. fit the novelty index, calibrate it, and export it pickle-free ──
    index_pkl = run_dir / "knn_index.pkl"
    stage(
        "knn_fit",
        knn_ood.cli_fit,
        ns(
            embeddings=str(inference_npz),
            labels=str(final_metadata),
            output=str(index_pkl),
            k=knn_k,
            distance_metric="cosine",
            sep="|",
        ),
        skip_if=maybe([index_pkl]),
    )

    id_distances = run_dir / "knn_id_distances.csv"
    stage(
        "knn_predict_id",
        knn_ood.cli_predict,
        ns(
            input_type="id",
            embeddings=str(inference_npz),
            labels=str(final_metadata),
            knn_index=str(index_pkl),
            output=str(id_distances),
            threshold_percentile=threshold_percentile,
            max_k=1,
            topk_output=None,
            k_grid=None,
            k_grid_output=None,
            sep="|",
        ),
        skip_if=maybe([id_distances]),
    )

    index_npz = run_dir / "knn_index.npz"
    stage(
        "knn_export",
        knn_ood.cli_export,
        ns(
            knn_index=str(index_pkl),
            output=str(index_npz),
            threshold_percentile=threshold_percentile,
            config_output=None,
        ),
        skip_if=maybe([index_npz]),
    )

    manifest_out = {
        "model": str(checkpoint),
        "knn_index": str(index_npz),
        "energy_summary": str(energy_summary) if energy_summary.is_file() else None,
        "device": device,
        "epochs": epochs,
        "n_samples": len(samples),
        "model_params": params,
        "threshold_percentile": threshold_percentile,
        "knn_k": knn_k,
    }
    (run_dir / "allcaps_run.json").write_text(json.dumps(manifest_out, indent=2))

    logger.info(
        "Training complete. Serotype with this model using:\n"
        "    ALLCAPS predict --input <samples> --extract align \\\n"
        "        --model %s \\\n"
        "        --knn-index %s \\\n"
        "        --energy-summary %s \\\n"
        "        --output <dir>",
        checkpoint, index_npz, energy_summary,
    )
    return manifest_out
