"""``ALLCAPS predict`` — assemblies in, serotype calls out."""

import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from ..consts import (
    CONTIG_SEP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_HEAD_MODEL,
    DEFAULT_MAX_LEN,
    DEFAULT_MODEL,
    DEFAULT_STRIDE_RATIO,
)
from ..logging_config import get_logger
from . import artifacts as art
from ._invoke import ns, resolve_device, stage
from .manifest import Sample

logger = get_logger(__name__)

#: Rolling-window step used by `--extract scan` (process_trihead_query's own default).
DEFAULT_SCAN_STEP = 2000
#: Novelty operating point. 95th percentile of the training leave-one-out 1-NN distances.
DEFAULT_THRESHOLD_PERCENTILE = 95.0
DEFAULT_MAX_K = 5
DEFAULT_ENERGY_PERCENTILE = 93.0

#: Columns of the final table, in order.
OUTPUT_COLUMNS = [
    "sample",
    "contig",
    "is_cps",
    "serotype",
    "serotype_confidence",
    "genogroup",
    "is_novel_knn",
    "nn_serotype",
    "nn_distance",
    "is_novel_energy",
    "energy",
    "note",
]


# ──────────────────────────────────────────────────────────────
#  Input preparation
# ──────────────────────────────────────────────────────────────


def _cut_loci(samples: List[Sample], work: Path, flanks: Path, cutoff: float,
              max_extension: int) -> Path:
    """Run the aligner-based cutter; returns the FASTA of cut loci."""
    from .. import data_locus_cutter

    infiles = work / "infiles.txt"
    infiles.write_text("".join(f"{s.path}\n" for s in samples))

    outpref = work / "cut"
    stage(
        "extract (align)",
        data_locus_cutter.main,
        ns(
            data_locus_cutter.build_parser,
            infiles=str(infiles),
            query=str(flanks),
            cutoff=cutoff,
            outpref=str(outpref),
            max_extension=max_extension,
            save_noncbl=False,
        ),
    )

    partial = Path(f"{outpref}_partial.txt")
    if partial.is_file() and partial.read_text().strip():
        n = len(partial.read_text().split())
        logger.warning(
            "%d assembly/assemblies had only one of the two flanking genes. Their locus "
            "was cut from that single flank (see the `cut=` field in the FASTA "
            "description), so the call rests on a partial locus.", n,
        )

    cut = Path(f"{outpref}.fasta")
    if not cut.is_file() or cut.stat().st_size == 0:
        absent = Path(f"{outpref}_absent.txt")
        n_absent = len(absent.read_text().split()) if absent.is_file() else 0
        raise RuntimeError(
            f"No cps locus was cut from any of the {len(samples)} input assemblies "
            f"({n_absent} had no dexB/aliA hit at all).\n"
            f"Check that the inputs are S. pneumoniae assemblies, try a lower "
            f"--cutoff (currently {cutoff}), or use --extract scan to search the "
            f"assembly directly without relying on the flanking genes."
        )
    return cut


def _concat_for_scan(samples: List[Sample], work: Path) -> Path:
    """Concatenate inputs into one FASTA, renaming records `Public_ID#Contig_ID`.

    Scan mode skips the cutter, so the record ids have to be built here to keep the
    rest of the pipeline on one id convention.
    """
    from Bio import SeqIO

    from ..data_locus_cutter import file_handler, safe_contig_id

    out = work / "scan_input.fasta"
    total_bp = 0
    with open(out, "w") as fh:
        for sample in samples:
            opener = file_handler(str(sample.path))
            with opener(str(sample.path)) as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    rid = f"{sample.sample_id}{CONTIG_SEP}{safe_contig_id(record.id)}"
                    fh.write(f">{rid}\n{str(record.seq)}\n")
                    total_bp += len(record.seq)

    windows = max(0, total_bp // DEFAULT_SCAN_STEP)
    logger.warning(
        "Scan mode will embed roughly %s rolling windows across %.1f Mbp. Each window "
        "is a %d bp slice chunked for ProkBERT, so this is far slower than "
        "--extract align — hours per genome on CPU. Use align unless the flanking "
        "genes are genuinely absent.",
        f"{windows:,}", total_bp / 1e6, DEFAULT_MAX_LEN,
    )
    return out


# ──────────────────────────────────────────────────────────────
#  Model + novelty
# ──────────────────────────────────────────────────────────────


def _run_query(query_fasta: Path, work: Path, artifacts: art.Artifacts, device: str,
               inference_mode: str, base_model: str, scan_step: int,
               energy_percentile: float) -> Path:
    """Serotype calls + pooled embeddings. Writes into ``work``."""
    from ..trihead import process_trihead_query

    model_params = {
        "max_length": DEFAULT_MAX_LEN,
        "rolling_step": scan_step,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "stride_ratio": DEFAULT_STRIDE_RATIO,
    }
    stage(
        "serotype",
        process_trihead_query.main,
        ns(
            process_trihead_query.build_parser,
            query=str(query_fasta),
            output_dir=str(work),
            device=device,
            base_model=base_model,
            head_model=DEFAULT_HEAD_MODEL,
            model_path=str(artifacts.model),
            model_params=model_params,
            inference_mode=inference_mode,
            energy_percentile=energy_percentile,
            energy_summary=(
                str(artifacts.energy_summary) if artifacts.energy_summary else None
            ),
        ),
    )
    return work / "query_results.csv"


def _run_knn(work: Path, artifacts: art.Artifacts, threshold_percentile: float,
             max_k: int) -> Optional[Path]:
    """The deployed novelty call. None when no index is available."""
    if not artifacts.can_score_knn:
        return None

    from .. import knn_ood

    out = work / "knn_query_distances.csv"
    stage(
        "novelty (kNN)",
        knn_ood.cli_predict,
        ns(
            input_type="query",
            embeddings=str(work / "query_embeddings.npz"),
            labels=None,
            knn_index=str(artifacts.knn_index),
            output=str(out),
            threshold_percentile=threshold_percentile,
            max_k=max_k,
            topk_output=None,
            k_grid=None,
            k_grid_output=None,
            sep="|",
        ),
    )
    return out


# ──────────────────────────────────────────────────────────────
#  Output assembly
# ──────────────────────────────────────────────────────────────


def _merge(query_results: Path, knn_distances: Optional[Path]):
    """Join the serotype and novelty tables into the user-facing one."""
    import pandas as pd

    # Serotype labels are strings ("19A", "15B/15C"); letting pandas infer turns a
    # purely numeric batch into floats and prints serotype 1 as "1.0".
    q = pd.read_csv(
        query_results,
        index_col=0,
        dtype={"pred_argmax": "string", "pred_genogroup": "string"},
    )
    q.index.name = "record_id"
    q = q.reset_index()

    # Record ids are `Public_ID#Contig_ID`; split back out for readability.
    split = q["record_id"].astype(str).str.split(CONTIG_SEP, n=1, expand=True)
    q["sample"] = split[0]
    q["contig"] = split[1] if split.shape[1] > 1 else ""

    q = q.rename(
        columns={
            "is_cbl": "is_cps",
            "pred_argmax": "serotype",
            "pred_genogroup": "genogroup",
            "novelty_confidence": "energy",
        }
    )

    if knn_distances is not None and Path(knn_distances).is_file():
        k = pd.read_csv(knn_distances, dtype={"nn_serotype": "string"})
        # `knn_distance` is the distance the novelty decision is made on (to the k-th
        # neighbour); `nn_distance` is to the closest one, and so is the value that
        # pairs with nn_serotype. At the deployed k=1 they are the same number.
        keep = [c for c in ("sample_id", "is_novel_knn", "nn_serotype", "nn_distance")
                if c in k.columns]
        q = q.merge(
            k[keep].rename(columns={"sample_id": "record_id"}),
            on="record_id",
            how="left",
        )

    for col in OUTPUT_COLUMNS:
        if col not in q.columns:
            q[col] = pd.NA

    q["note"] = [_note(row) for _, row in q.iterrows()]

    ordered = ["record_id"] + OUTPUT_COLUMNS
    return q[ordered]


def _note(row) -> str:
    """Say plainly how much weight the serotype call carries.

    A confident serotype printed next to a contig the model does not think is a cps
    locus, or one it flags as novel, invites exactly the wrong reading.
    """
    import pandas as pd

    notes = []
    is_cps = row.get("is_cps")
    if is_cps is not None and not pd.isna(is_cps) and not bool(is_cps):
        notes.append("not a cps locus - serotype call not meaningful")
    if bool(row.get("is_novel_knn") is True):
        nn = row.get("nn_serotype")
        nn_txt = "" if pd.isna(nn) else f"; nearest known: {nn}"
        notes.append(f"NOVEL - unlike any training serotype{nn_txt}")
    if pd.isna(row.get("is_novel_knn")):
        notes.append("novelty not assessed (no kNN index)")
    return "; ".join(notes)


def _add_unresolved(table, samples: List[Sample]):
    """Append a row for every input that produced no locus.

    A batch run that silently returns fewer rows than it was given is how a sample
    goes missing unnoticed, so absent inputs are stated rather than dropped.
    """
    import pandas as pd

    got = set(table["sample"].astype(str))
    missing = [s for s in samples if s.sample_id not in got]
    if not missing:
        return table

    logger.warning(
        "%d of %d input sample(s) yielded no cps locus: %s",
        len(missing), len(samples),
        ", ".join(s.sample_id for s in missing[:5])
        + ("..." if len(missing) > 5 else ""),
    )
    rows = pd.DataFrame(
        [
            {
                "record_id": s.sample_id,
                "sample": s.sample_id,
                "contig": pd.NA,
                "note": "no cps locus found - no dexB/aliA hit above --cutoff",
            }
            for s in missing
        ]
    )
    out = pd.concat([table, rows], ignore_index=True)
    return out[["record_id"] + OUTPUT_COLUMNS]


def _render(df) -> str:
    """A plain aligned table for stdout."""
    import pandas as pd

    show = df.drop(columns=["record_id"])
    with pd.option_context("display.max_rows", None, "display.width", 200):
        return show.to_string(index=False, na_rep="-")


# ──────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────


def run(
    samples: List[Sample],
    extract: str,
    output: Optional[Path],
    model: Optional[str] = None,
    knn_index: Optional[str] = None,
    energy_summary: Optional[str] = None,
    flanks: Optional[str] = None,
    device: str = "auto",
    base_model: str = DEFAULT_MODEL,
    cutoff: float = 0.7,
    max_extension: int = 30_000,
    scan_step: int = DEFAULT_SCAN_STEP,
    threshold_percentile: float = DEFAULT_THRESHOLD_PERCENTILE,
    energy_percentile: float = DEFAULT_ENERGY_PERCENTILE,
    max_k: int = DEFAULT_MAX_K,
    hf_repo: str = art.DEFAULT_HF_REPO,
    revision: Optional[str] = None,
    offline: bool = False,
    quiet: bool = False,
    keep_intermediates: bool = False,
):
    """Assemblies -> one table of serotype and novelty calls."""
    device = resolve_device(device)
    logger.info("Using device: %s", device)

    artifacts = art.resolve_artifacts(
        model=model,
        knn_index=knn_index,
        energy_summary=energy_summary,
        repo=hf_repo,
        revision=revision,
        offline=offline,
    )

    out_dir = Path(output).expanduser() if output else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        work = out_dir / "intermediates"
        work.mkdir(exist_ok=True)
        tmp_work = None
    else:
        tmp_work = tempfile.mkdtemp(prefix="allcaps-")
        work = Path(tmp_work)

    try:
        if extract == "align":
            flank_path = Path(flanks).expanduser() if flanks else art.packaged_flanks()
            query_fasta = _cut_loci(samples, work, flank_path, cutoff, max_extension)
            inference_mode = "eval"
        else:
            query_fasta = _concat_for_scan(samples, work)
            inference_mode = "scan"

        query_results = _run_query(
            query_fasta, work, artifacts, device, inference_mode, base_model, scan_step,
            energy_percentile,
        )
        knn_distances = _run_knn(work, artifacts, threshold_percentile, max_k)
        table = _merge(query_results, knn_distances)
        table = _add_unresolved(table, samples)

        if out_dir:
            dest = out_dir / "predictions.tsv"
            table.to_csv(dest, sep="\t", index=False)
            logger.info("Wrote %d prediction(s) to %s", len(table), dest)
            topk = work / "knn_query_distances_topk.csv"
            if topk.is_file():
                shutil.copy(topk, out_dir / "neighbours_topk.csv")

        if not quiet:
            print(_render(table))

        return table
    finally:
        if tmp_work and not keep_intermediates:
            shutil.rmtree(tmp_work, ignore_errors=True)
        elif out_dir and not keep_intermediates:
            shutil.rmtree(work, ignore_errors=True)
