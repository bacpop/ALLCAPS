"""Query processing pipeline for the trihead transformer serotyping model.

Processes FASTA query sequences through the full two-stage pipeline:

1. ProkBERT base model  → chunk embeddings
2. TransformerTriHeadLR → CBL, serotype, genogroup logits + z embedding

Supports two inference modes:

- ``eval``: Single window at position 0 (matches training pipeline exactly).
- ``scan``: Rolling window with mean-logit aggregation (for novel / full-genome
  queries).

Usage::

    python -m scripts.trihead.process_trihead_query \\
        --query query.fasta \\
        --model_path results/transformer_model.pth \\
        --output_dir results/ \\
        --inference_mode eval
"""

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from tqdm import tqdm

from ..consts import DEFAULT_ENERGY_TEMPERATURE, DEFAULT_MAX_LEN, DEFAULT_MODEL
from ..inference import (
    embed_sequence,
    energy_score,
    load_base_model,
    load_trained_model,
    parse_model_params_json,
    set_deterministic_seeds,
    softmax_predict,
)
from ..openmax import OpenMax

THRESH_CPS = 0.5
DEFAULT_ENERGY_PERCENTILE = 99.0
DEFAULT_ROLLING_STEP = 2000

# Temporary hard-coded energy thresholds (TODO: load from JSON / CLI)
_PERCENTILES_SEROTYPE = {
    "93.0": -8.152,
    "95.0": -8.368741035461426,
    "99.0": -6.334590911865234,
    "99.5": -5.951267242431641,
}


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────


def main(args):
    device = args.device
    max_length = args.model_params.get("max_length", DEFAULT_MAX_LEN)
    rolling_step = args.model_params.get("rolling_step", DEFAULT_ROLLING_STEP)
    cbl_threshold = THRESH_CPS

    # Energy threshold
    resolved_temperature = float(args.energy_temperature)
    tau_serotype: Optional[float] = _PERCENTILES_SEROTYPE.get(
        str(args.energy_percentile)
    )
    assert tau_serotype is not None, "tau_serotype must be set"

    set_deterministic_seeds()

    # ── Load models via shared inference module ───────────────
    base_kwargs: dict = {"model_name": args.base_model, "device": device}
    if "chunk_size" in args.model_params:
        base_kwargs["chunk_size"] = args.model_params["chunk_size"]
    if "stride_ratio" in args.model_params:
        base_kwargs["stride_ratio"] = args.model_params["stride_ratio"]

    print(f"Loading the {args.base_model} base model...")
    base_bundle = load_base_model(**base_kwargs)

    print("Loading the transformer and logistic regression model...")
    head_bundle = load_trained_model(args.model_path, device, args.head_model)
    logistic_model = head_bundle.model
    idx_to_serotype = head_bundle.idx_to_serotype
    idx_to_genogroup = head_bundle.idx_to_genogroup

    # Load OpenMax if provided
    openmax_model: Optional[OpenMax] = None
    if args.openmax_params and os.path.isfile(args.openmax_params):
        print(f"Loading OpenMax parameters from {args.openmax_params}")
        openmax_model = OpenMax.load(args.openmax_params)

    # ── Process each query sequence ───────────────────────────
    print("Processing queries...")
    results: dict = {}
    query_sequences = list(SeqIO.parse(args.query, "fasta"))

    for record in tqdm(query_sequences, desc="Processing queries"):
        print(f"Processing query: {record.id}...")

        # Canonical embedding — single source of truth via inference.embed_sequence
        cbl_list, sero_list, geno_list, z_list = embed_sequence(
            base_bundle=base_bundle,
            head_model=logistic_model,
            sequence=str(record.seq),
            device=device,
            max_length=max_length,
            inference_mode=args.inference_mode,
            scan_step=rolling_step,
        )

        # ── Window aggregation ────────────────────────────────
        n_windows = len(sero_list)

        if n_windows == 1:
            sel_cbl = np.asarray(cbl_list[0]).squeeze()
            sel_sero = np.asarray(sero_list[0]).squeeze()
            sel_geno = (
                None
                if geno_list is None
                else np.asarray(geno_list[0]).squeeze()
            )
            sel_z = np.asarray(z_list[0])
        else:
            # Scan mode: mean-logit pooling over capsulated windows
            capsulated = []
            for wi in range(n_windows):
                cbl_vec = np.asarray(cbl_list[wi]).squeeze()
                cbl_probs = torch.softmax(
                    torch.tensor(cbl_vec, dtype=torch.float32), dim=-1
                ).numpy()
                if cbl_probs[1] > cbl_threshold:
                    capsulated.append(wi)

            use = capsulated if capsulated else list(range(n_windows))

            sel_cbl = np.mean(
                [np.asarray(cbl_list[i]).squeeze() for i in use], axis=0
            )
            sel_sero = np.mean(
                [np.asarray(sero_list[i]).squeeze() for i in use], axis=0
            )
            sel_geno = None
            if geno_list is not None:
                sel_geno = np.mean(
                    [np.asarray(geno_list[i]).squeeze() for i in use], axis=0
                )
            sel_z = np.mean(
                [np.asarray(z_list[i]) for i in use], axis=0
            )

        # ── Predictions ───────────────────────────────────────
        cbl_probs = torch.softmax(
            torch.tensor(sel_cbl, dtype=torch.float32), dim=-1
        ).numpy()
        is_cbl = bool(cbl_probs[1] > cbl_threshold)

        _, sero_conf, _ = softmax_predict(sel_sero, idx_to_serotype)

        # Energy novelty
        e_sero = energy_score(sel_sero, temperature=resolved_temperature)
        is_novel_energy = bool(e_sero > tau_serotype)

        # OpenMax novelty
        openmax_pred = None
        openmax_prob_unknown = None
        is_novel_openmax = False
        if openmax_model is not None:
            om = openmax_model.openmax_score(sel_z, sel_sero)
            openmax_pred = om["openmax_pred"]
            openmax_prob_unknown = round(om["prob_unknown"], 6)
            is_novel_openmax = om["is_novel"]

        is_novel = is_novel_energy or is_novel_openmax

        entry: dict = {
            "serotype_logits": sel_sero,
            "embedding": sel_z,
            "is_cbl": is_cbl,
            "is_novel_serogroup": is_novel,
            "is_novel_energy": is_novel_energy,
            "is_novel_openmax": is_novel_openmax,
            "serotype_confidence": round(sero_conf, 3),
            "novelty_confidence": round(e_sero, 3),
        }
        if openmax_pred is not None:
            entry["openmax_pred"] = openmax_pred
            entry["openmax_prob_unknown"] = openmax_prob_unknown
        if sel_geno is not None and idx_to_genogroup is not None:
            geno_label, _, _ = softmax_predict(sel_geno, idx_to_genogroup)
            entry["genogroup_logits"] = sel_geno
            entry["pred_genogroup"] = geno_label

        results[record.id] = entry

    # ── Save results ──────────────────────────────────────────
    results_df = pd.DataFrame.from_dict(results, orient="index")
    results_df["pred_argmax"] = results_df["serotype_logits"].apply(
        lambda x: softmax_predict(x, idx_to_serotype)[0]
    )

    drop_cols = ["embedding", "serotype_logits"]
    if "genogroup_logits" in results_df.columns:
        results_df["pred_genogroup"] = results_df["genogroup_logits"].apply(
            lambda x: softmax_predict(x, idx_to_genogroup)[0]
        )
        drop_cols.append("genogroup_logits")

    os.makedirs(args.output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.output_dir, "query_embeddings.npz"),
        record_ids=results_df.index.to_numpy(),
        embeddings=np.stack(results_df["embedding"].values),
    )
    results_df.drop(columns=drop_cols).to_csv(
        os.path.join(args.output_dir, "query_results.csv")
    )


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Query processing via trihead serotyping model."
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"]
    )
    p.add_argument(
        "--query", type=str, required=True,
        help="Path to the query FASTA file.",
    )
    p.add_argument("--model_params", type=str, default="{}")
    p.add_argument("--base_model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--head_model", type=str, default="transformer_trihead_lr")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument(
        "--energy_temperature",
        type=float,
        default=DEFAULT_ENERGY_TEMPERATURE,
    )
    p.add_argument(
        "--energy_percentile",
        type=float,
        default=DEFAULT_ENERGY_PERCENTILE,
    )
    p.add_argument(
        "--query_mode",
        default="default",
        choices=["default", "fast"],
    )
    p.add_argument(
        "--inference_mode",
        default="eval",
        choices=["eval", "scan"],
        help="'eval' matches training; 'scan' uses rolling window.",
    )
    p.add_argument(
        "--openmax_params",
        type=str,
        default=None,
        help="Path to fitted OpenMax .pkl for open-set recognition.",
    )

    args = p.parse_args()
    args.model_params = parse_model_params_json(args.model_params)
    return args


if __name__ == "__main__":
    main(parse_args())

