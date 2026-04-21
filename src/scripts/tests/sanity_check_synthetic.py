#!/usr/bin/env python
"""
Synthetic FASTA sanity check:
  Generates synthetic sequences to validate model behaviour on edge-cases:
    (a) Random DNA — should be flagged as novel / low confidence.
    (b) Chimeric sequences — concatenation of chunks from 2 different serotypes.
    (c) Truncated real sequences — half-length of a real capsule locus.

  Runs them through process_trihead_query.py (eval + scan modes) and reports
  confidence / energy / CBL / novelty flags.

Usage (from repo root):
    python -m scripts.tests.sanity_check_synthetic \
        --fasta  data/GPS_All_S_pneumoniae_CBL_cleaned.fasta \
        --labels results/final_metadata.tsv \
        --model  results/transformer_model.pth \
        --output_dir results/sanity_synthetic \
        --device cpu
"""

import argparse
import os
import random
import tempfile

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import torch

from ..consts import (
    DEFAULT_MODEL, DEFAULT_CHUNK_SIZE, DEFAULT_MAX_LEN,
    DEFAULT_STRIDE_RATIO
)
from ..logging_config import get_logger
from ..models import ModelRegistry

logger = get_logger(__name__)


# ───────────────────────────── Generators ─────────────────────────────

def random_dna(length: int, seed: int = 0) -> str:
    """Generate a uniformly random DNA sequence."""
    rng = random.Random(seed)
    return "".join(rng.choices("ACGT", k=length))


def generate_random_records(n: int = 5, min_len: int = 20_000, max_len: int = 30_000, seed: int = 42):
    """Yield SeqRecords of purely random DNA."""
    rng = random.Random(seed)
    for i in range(n):
        length = rng.randint(min_len, max_len)
        seq = random_dna(length, seed=seed + i)
        yield SeqRecord(Seq(seq), id=f"random_{i}", description="purely random DNA")


def generate_chimeric_records(fasta_path: str, labels_path: str, n: int = 5, seed: int = 42):
    """
    Build chimeric sequences by joining the first half of one serotype's
    locus with the second half of a different serotype's locus.
    """
    labels_df = pd.read_csv(labels_path, sep="\t", index_col=0)
    capsulated = labels_df[labels_df["Is_capsule"].astype(bool) & (labels_df["Serotype"] != "Non-typeable")]
    rng = random.Random(seed)

    # Preload a subset of sequences keyed by public_id
    id_to_seq = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        pub_id = record.id.split("__")[0]
        if pub_id in capsulated.index and pub_id not in id_to_seq:
            id_to_seq[pub_id] = str(record.seq)
        if len(id_to_seq) >= 200:  # enough for chimeras
            break

    sero_to_ids = {}
    for pid, sero in zip(capsulated.index, capsulated["Serotype"]):
        if pid in id_to_seq:
            sero_to_ids.setdefault(sero, []).append(pid)

    valid_seros = [s for s in sero_to_ids if len(sero_to_ids[s]) >= 1]

    for i in range(n):
        s1, s2 = rng.sample(valid_seros, 2)
        id1 = rng.choice(sero_to_ids[s1])
        id2 = rng.choice(sero_to_ids[s2])
        seq1 = id_to_seq[id1]
        seq2 = id_to_seq[id2]
        mid1 = len(seq1) // 2
        mid2 = len(seq2) // 2
        chimera = seq1[:mid1] + seq2[mid2:]
        yield SeqRecord(
            Seq(chimera),
            id=f"chimera_{i}_{s1}_{s2}",
            description=f"chimera of {s1} ({id1}) and {s2} ({id2})"
        )


def generate_truncated_records(fasta_path: str, labels_path: str, n: int = 5, seed: int = 42):
    """Yield SeqRecords that are the first half of real capsule sequences."""
    labels_df = pd.read_csv(labels_path, sep="\t", index_col=0)
    capsulated = labels_df[labels_df["Is_capsule"].astype(bool) & (labels_df["Serotype"] != "Non-typeable")]
    rng = random.Random(seed)

    sampled = rng.sample(capsulated.index.tolist(), min(n * 3, len(capsulated)))
    count = 0
    for record in SeqIO.parse(fasta_path, "fasta"):
        pub_id = record.id.split("__")[0]
        if pub_id in sampled:
            seq = str(record.seq)
            half = seq[: len(seq) // 2]
            sero = capsulated.loc[pub_id, "Serotype"]
            if isinstance(sero, pd.Series):
                sero = sero.iloc[0]
            yield SeqRecord(
                Seq(half),
                id=f"trunc_{count}_{sero}_{pub_id}",
                description=f"truncated {pub_id} (serotype {sero})"
            )
            count += 1
            if count >= n:
                break


# ───────────────────────────── Main ─────────────────────────────

def run_pipeline(fasta_path, model_path, base_model_name, head_model, device,
                 chunk_size, stride_ratio, max_length, inference_mode="eval"):
    """Run query pipeline and return DataFrame with results."""
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    from ..trihead.process_trihead_query import transformer_embedding, energy_score

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    nt_model = AutoModelForMaskedLM.from_pretrained(base_model_name, trust_remote_code=True).to(device)
    nt_model.eval()

    model_save_dict = torch.load(model_path, map_location=device)
    model_config = model_save_dict['model_config']
    idx_to_serotype = {v: k for k, v in model_save_dict['serotype_to_idx'].items()}

    logistic_model = ModelRegistry.get_model_class(head_model).from_config(model_config).to(device)
    logistic_model.load_state_dict(model_save_dict['model_state_dict'])
    logistic_model.eval()

    rows = {}
    for record in tqdm(SeqIO.parse(fasta_path, "fasta"), desc=f"Pipeline ({inference_mode})"):
        cbl_list, sero_list, _, emb_list = transformer_embedding(
            tokenizer=tokenizer,
            base_model=nt_model,
            logistic_model=logistic_model,
            sequence=str(record.seq),
            device=device,
            chunk_size=chunk_size,
            stride_ratio=stride_ratio,
            step=2000,
            max_length=max_length,
            inference_mode=inference_mode,
        )

        sel_cbl = np.asarray(cbl_list[0]).squeeze()
        sel_sero = np.asarray(sero_list[0]).squeeze()

        cbl_probs = torch.softmax(torch.tensor(sel_cbl, dtype=torch.float32), dim=-1).numpy()
        is_cbl = bool(cbl_probs[1] > 0.5)
        sero_probs = torch.softmax(torch.tensor(sel_sero, dtype=torch.float32), dim=-1).numpy()
        pred_idx = int(np.argmax(sero_probs))
        pred_serotype = idx_to_serotype[pred_idx]
        max_conf = float(sero_probs.max())
        energy = energy_score(sel_sero, temperature=1.0)

        rows[record.id] = {
            "pred_serotype": pred_serotype,
            "is_cbl": is_cbl,
            "max_confidence": round(max_conf, 4),
            "energy": round(energy, 4),
            "description": record.description,
        }

    return pd.DataFrame.from_dict(rows, orient="index")


def main():
    parser = argparse.ArgumentParser(description="Synthetic FASTA sanity check")
    parser.add_argument("--fasta", required=True, help="Original training FASTA (for chimeric/truncated)")
    parser.add_argument("--labels", required=True, help="Final metadata TSV")
    parser.add_argument("--model", required=True, help="Trained model .pth")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--base_model", default=DEFAULT_MODEL)
    parser.add_argument("--head_model", default="transformer_trihead_lr")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--stride_ratio", type=float, default=DEFAULT_STRIDE_RATIO)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--n_samples", type=int, default=5, help="Number of synthetic samples per category")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(42)

    # ── Generate synthetic FASTA ──
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".fasta", mode="w")
    records = []

    logger.info("Generating random DNA sequences...")
    for rec in generate_random_records(n=args.n_samples):
        records.append(rec)

    logger.info("Generating chimeric sequences...")
    for rec in generate_chimeric_records(args.fasta, args.labels, n=args.n_samples):
        records.append(rec)

    logger.info("Generating truncated sequences...")
    for rec in generate_truncated_records(args.fasta, args.labels, n=args.n_samples):
        records.append(rec)

    SeqIO.write(records, tmp, "fasta")
    tmp.close()
    logger.info("Wrote %d synthetic records to %s", len(records), tmp.name)

    # ── Run pipeline ──
    logger.info("=== Running query pipeline (eval mode) ===")
    results_df = run_pipeline(
        fasta_path=tmp.name,
        model_path=args.model,
        base_model_name=args.base_model,
        head_model=args.head_model,
        device=args.device,
        chunk_size=args.chunk_size,
        stride_ratio=args.stride_ratio,
        max_length=args.max_length,
        inference_mode="eval",
    )

    # ── Tag categories ──
    results_df["category"] = results_df.index.map(
        lambda x: "random" if x.startswith("random_")
        else "chimera" if x.startswith("chimera_")
        else "truncated" if x.startswith("trunc_")
        else "other"
    )

    # ── Report ──
    report_lines = [
        "=" * 60,
        "SYNTHETIC FASTA SANITY CHECK REPORT",
        "=" * 60,
        "",
    ]

    for cat in ["random", "chimera", "truncated"]:
        subset = results_df[results_df["category"] == cat]
        if len(subset) == 0:
            continue
        report_lines.append(f"--- {cat.upper()} ({len(subset)} samples) ---")
        report_lines.append(f"  Mean confidence: {subset['max_confidence'].mean():.4f}")
        report_lines.append(f"  Mean energy:     {subset['energy'].mean():.4f}")
        report_lines.append(f"  CBL rate:        {subset['is_cbl'].mean():.4f}")
        report_lines.append(f"  Unique preds:    {subset['pred_serotype'].nunique()}")
        report_lines.append("")
        for idx in subset.index:
            row = subset.loc[idx]
            report_lines.append(
                f"  {idx}: pred={row['pred_serotype']}, conf={row['max_confidence']:.3f}, "
                f"energy={row['energy']:.3f}, cbl={row['is_cbl']}"
            )
        report_lines.append("")

    # Expectations
    report_lines.append("=== EXPECTATIONS ===")
    random_subset = results_df[results_df["category"] == "random"]
    if len(random_subset) > 0:
        high_energy = (random_subset["energy"] > -8.0).mean()
        low_conf = (random_subset["max_confidence"] < 0.5).mean()
        report_lines.append(f"Random: {high_energy:.0%} have high energy (>-8), {low_conf:.0%} have low confidence (<0.5)")
        if high_energy >= 0.8:
            report_lines.append("  PASS: Most random sequences have high energy (would be novel)")
        else:
            report_lines.append("  INFO: Random sequences may not be detected as novel — energy threshold may need tuning")

    chimera_subset = results_df[results_df["category"] == "chimera"]
    if len(chimera_subset) > 0:
        mean_conf = chimera_subset["max_confidence"].mean()
        report_lines.append(f"Chimera: mean confidence = {mean_conf:.4f}")
        report_lines.append("  INFO: Chimeric sequences expected to have degraded confidence")

    report_text = "\n".join(report_lines)
    logger.info("\n%s", report_text)

    report_path = os.path.join(args.output_dir, "synthetic_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    results_df.to_csv(os.path.join(args.output_dir, "synthetic_results.csv"))

    os.unlink(tmp.name)
    logger.info("Report saved to %s", report_path)


if __name__ == "__main__":
    main()
