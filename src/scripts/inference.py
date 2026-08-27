"""Shared inference utilities for the pneumococcal serotyping pipeline.

This module centralises model-loading, sequence-embedding, scoring, and
data I/O functions that were previously duplicated across
process_trihead_query.py, eval_serotype_classifier.py and
eval_cbl_classifier.py.

Typical usage
-------------
    from scripts.inference import (
        load_trained_model, load_base_model, embed_sequence,
        energy_score, set_deterministic_seeds,
    )

    set_deterministic_seeds()
    base = load_base_model("neuralbioinfo/prokbert-mini-long", device="cuda")
    head = load_trained_model("model.pth", device="cuda")
    cbl, sero, geno, z = embed_sequence(base, head.model, sequence, device="cuda")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

from .consts import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_LEN,
    DEFAULT_MISSING_LABEL,
    DEFAULT_MODEL,
    DEFAULT_SEP,
    DEFAULT_STRIDE_RATIO,
)
from .logging_config import get_logger
from .models import ModelRegistry
from .utils import chunk_sequence, embed_chunks, get_sample_id

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Data-classes
# ═══════════════════════════════════════════════════════════════


@dataclass
class TrainedModelBundle:
    """Everything needed from a saved ``.pth`` checkpoint."""

    model: torch.nn.Module
    config: dict
    serotype_to_idx: Dict[str, int]
    idx_to_serotype: Dict[int, str]
    genogroup_to_idx: Optional[Dict[str, int]] = None
    idx_to_genogroup: Optional[Dict[int, str]] = None
    num_serotypes: int = 0
    num_genogroups: int = 0


@dataclass
class BaseModelBundle:
    """Tokenizer + pretrained base LM ready for inference."""

    tokenizer: object  # AutoTokenizer (deferred import)
    model: torch.nn.Module  # AutoModel (encoder only)
    model_max_length: int
    chunk_size: int
    stride: int


# ═══════════════════════════════════════════════════════════════
#  Deterministic seeds
# ═══════════════════════════════════════════════════════════════


def set_deterministic_seeds(seed: int = 42) -> None:
    """Set PyTorch seeds for reproducible inference."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════


def load_trained_model(
    path: str,
    device: Union[str, torch.device] = "cpu",
    head_model: str = "transformer_trihead_lr",
) -> TrainedModelBundle:
    """Load a saved ``.pth`` checkpoint and return a ready-to-use bundle.

    The returned ``model`` is already in ``eval()`` mode on *device*.
    """
    device = torch.device(device)
    sd = torch.load(path, map_location=device)
    cfg = sd["model_config"]
    s2i = sd["serotype_to_idx"]
    g2i = sd.get("genogroup_to_idx")

    model = ModelRegistry.get_model_class(head_model).from_config(cfg).to(device)
    model.load_state_dict(sd["model_state_dict"])
    model.eval()

    return TrainedModelBundle(
        model=model,
        config=cfg,
        serotype_to_idx=s2i,
        idx_to_serotype={v: k for k, v in s2i.items()},
        genogroup_to_idx=g2i,
        idx_to_genogroup={v: k for k, v in g2i.items()} if g2i else None,
        num_serotypes=sd.get("num_serotypes", len(s2i)),
        num_genogroups=sd.get("num_genogroups", len(g2i) if g2i else 0),
    )


def load_base_model(
    model_name: str = DEFAULT_MODEL,
    device: Union[str, torch.device] = "cpu",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    stride_ratio: float = DEFAULT_STRIDE_RATIO,
) -> BaseModelBundle:
    """Load a pretrained foundation model (e.g. ProkBERT) for base embeddings.

    Returns a bundle with pre-calculated ``chunk_size`` and ``stride``
    already clipped to the tokenizer's ``model_max_length``.
    """
    from transformers import AutoModel, AutoTokenizer

    device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    mml = getattr(tokenizer, "model_max_length", chunk_size)
    cs = min(chunk_size, mml)

    return BaseModelBundle(
        tokenizer=tokenizer,
        model=model,
        model_max_length=mml,
        chunk_size=cs,
        stride=int(cs * stride_ratio),
    )


# ═══════════════════════════════════════════════════════════════
#  Scoring utilities
# ═══════════════════════════════════════════════════════════════


def energy_score(logits, temperature: float = 1.0) -> float:
    """Free-energy OOD score: ``E = −T · logsumexp(logits / T)``.

    Accepts both ``np.ndarray`` and ``torch.Tensor``.
    """
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    return float(
        (-temperature * torch.logsumexp(logits.float() / temperature, dim=-1)).item()
    )


def softmax_predict(
    logits, idx_to_label: Dict[int, str]
) -> Tuple[str, float, np.ndarray]:
    """Return ``(predicted_label, confidence, probabilities)`` from raw logits."""
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    probs = torch.softmax(logits.float(), dim=-1)
    conf, idx = probs.max(dim=-1)
    return idx_to_label[int(idx.item())], float(conf.item()), probs.numpy()


# ═══════════════════════════════════════════════════════════════
#  Full-sequence embedding  (query / inference pipeline)
# ═══════════════════════════════════════════════════════════════


def embed_sequence(
    base_bundle: BaseModelBundle,
    head_model: torch.nn.Module,
    sequence: str,
    device: Union[str, torch.device] = "cpu",
    max_length: int = DEFAULT_MAX_LEN,
    inference_mode: str = "eval",
    scan_step: int = 2000,
) -> Tuple[
    List[np.ndarray], List[np.ndarray], Optional[List[np.ndarray]], List[np.ndarray]
]:
    """Canonical  raw DNA → (cbl_logits, sero_logits, geno_logits|None, z).

    This is the **single source of truth** for converting a raw nucleotide
    string into logits and projected embeddings through the full two-stage
    model (ProkBERT base → TransformerTriHead / LR classifier).

    The function replicates the exact data-processing path used during
    training (embed_transformer.py → MultidomainChunkedDataset → model.forward):

    1. Truncate sequence to ``max_length`` from the chosen start position.
    2. Chunk via ``chunk_sequence(candidate, chunk_size, stride)``.
    3. Embed chunks through base LM (``embed_chunks``).
    4. Feed chunk-embeddings through the full head model forward pass
       (positional embedding → TransformerEncoder → masked mean-pool →
       MLP projection → L2-normalise → classifier heads).

    Parameters
    ----------
    base_bundle : ``BaseModelBundle`` from :func:`load_base_model`.
    head_model  : The trained head module (``TransformerTriHeadLR``, etc.).
    sequence    : Raw nucleotide string.
    device      : Torch device string or object.
    max_length  : Truncation window length (default matches
                  ``embed_transformer.py --seq_max_len``).
    inference_mode :
        ``"eval"``  - single window at position 0 (matches training).
        ``"scan"``  - rolling window with *scan_step* for novel queries.
    scan_step : Step between consecutive windows in ``"scan"`` mode.

    Returns
    -------
    cbl_logits_list, sero_logits_list, geno_logits_or_None, z_list
        Each list has one entry per sliding-window position.
    """
    device = torch.device(device)
    tok = base_bundle.tokenizer
    base = base_bundle.model
    cs = base_bundle.chunk_size
    st = base_bundle.stride
    mml = base_bundle.model_max_length

    # Determine start positions
    if inference_mode == "eval":
        starts = [0]
    else:
        starts = [
            i
            for i in range(0, len(sequence), scan_step)
            if i + max_length <= len(sequence)
        ]
        if not starts:
            starts = [0]

    all_cbl: List[np.ndarray] = []
    all_sero: List[np.ndarray] = []
    all_geno: Optional[List[np.ndarray]] = []
    all_z: List[np.ndarray] = []

    for s in starts:
        candidate = sequence[s : s + max_length]
        chunks = chunk_sequence(candidate, cs, st)
        if not chunks:
            chunks = [candidate]
        pooled = embed_chunks(chunks, tok, base, str(device), mml)

        with torch.no_grad():
            out = head_model(pooled.unsqueeze(0).to(device))

        # Handle both trihead (nested tuple) and 2-head models
        if isinstance(out[1], (list, tuple)) and len(out[1]) == 2:
            cbl, (sero, geno), z = out
            all_geno.append(geno.cpu().numpy())
        else:
            cbl, sero, z = out
            all_geno = None

        all_cbl.append(cbl.cpu().numpy())
        all_sero.append(sero.cpu().numpy())
        all_z.append(z.squeeze(0).cpu().numpy())

    return all_cbl, all_sero, all_geno, all_z


# ═══════════════════════════════════════════════════════════════
#  Classify from pre-computed z (eval scripts)
# ═══════════════════════════════════════════════════════════════


def classify_from_z(
    model: torch.nn.Module,
    z_embeddings: np.ndarray,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Run classifier heads on pre-computed z-vectors (from ``.npz``).

    This matches the eval path used by eval_serotype_classifier.py and
    eval_cbl_classifier.py.  The input z-vectors are the L2-normalised
    projected embeddings stored by infer_trihead_transformer.py.

    Returns ``(cbl_logits, sero_logits, geno_logits | None)``.
    """
    device = torch.device(device)
    all_cbl, all_sero, all_geno = [], [], []
    has_geno = hasattr(model, "genogroup_classifier")

    with torch.no_grad():
        for i in range(0, len(z_embeddings), batch_size):
            batch = torch.tensor(
                z_embeddings[i : i + batch_size],
                dtype=torch.float32,
                device=device,
            )
            all_cbl.append(model.cbl_classifier(batch).cpu().numpy())
            all_sero.append(model.serotype_classifier(batch).cpu().numpy())
            if has_geno:
                all_geno.append(model.genogroup_classifier(batch).cpu().numpy())

    return (
        np.concatenate(all_cbl),
        np.concatenate(all_sero),
        np.concatenate(all_geno) if has_geno else None,
    )


# ═══════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════


def load_labels(
    path: str,
    label_column: str = "Serotype",
    missing_label: str = DEFAULT_MISSING_LABEL,
    skip_labels: Optional[Sequence[str]] = None,
    labeled_only: bool = False,
) -> pd.DataFrame:
    """Load and clean a metadata TSV/CSV.

    Returns a filtered ``DataFrame`` indexed by sample ID with a
    normalised ``Serotype`` column.
    """
    sep = "\t" if path.endswith(".tsv") else ","
    df = pd.read_csv(path, sep=sep, index_col=0)
    df["Serotype"] = df[label_column].fillna(missing_label)
    if labeled_only:
        df = df[df["Serotype"] != missing_label]
    if skip_labels:
        df = df[~df["Serotype"].isin(list(skip_labels))]
    return df


def load_z_embeddings(
    npz_path: str,
    labels_df: pd.DataFrame,
    sep: str = DEFAULT_SEP,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load ``.npz`` z-embeddings aligned to a labels DataFrame.

    Returns ``(X_array, aligned_labels_df)`` with rows in the same order.
    """
    X = np.load(npz_path, allow_pickle=True)
    keys = (
        labels_df["Is_capsule"].map(lambda x: "cbl" if x else "non-cbl")
        + sep
        + get_sample_id(labels_df)
    )
    valid = [k for k in keys if k in X]
    mask = keys.isin(valid)
    labels_df = labels_df[mask.values].copy()
    keys = keys[mask.values]
    return np.stack([X[k] for k in keys]), labels_df


# ═══════════════════════════════════════════════════════════════
#  CLI helpers
# ═══════════════════════════════════════════════════════════════


def parse_model_params_json(raw: str) -> dict:
    """Parse a JSON string of model params; return ``{}`` on failure."""
    try:
        d = json.loads(raw)
        return d if isinstance(d, dict) else {}
    except json.JSONDecodeError:
        logger.error("Error parsing model parameters JSON string.")
        return {}


def parse_skip_labels(raw: str) -> List[str]:
    """Parse comma-separated skip-labels string."""
    try:
        return [s.strip() for s in raw.split(",") if s.strip()]
    except (ValueError, AttributeError):
        return []
