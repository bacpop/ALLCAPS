#!/usr/bin/env python
"""
Sanity-check round-trip test:
  1. Takes the training FASTA and metadata.
  2. Randomly samples N records per serotype.
  3. Writes them to a temporary FASTA.
  4. Runs the *fixed* process_trihead_query.py in inference_mode='eval' on that FASTA.
  5. Loads the inference .npz (from infer_trihead_transformer.py) for the same samples.
  6. Compares predicted serotype, CBL classification, and raw-logit cosine similarity.
  7. Reports per-sample match/mismatch and overall agreement rate.

Usage (from repo root):
    python -m scripts.tests.sanity_check_roundtrip \
        --fasta data/GPS_All_S_pneumoniae_CBL_cleaned.fasta \
        --labels results/final_metadata.tsv \
        --model  results/transformer_model.pth \
        --inference_npz results/inference_results.npz \
        --output_dir results/sanity_roundtrip \
        --samples_per_serotype 5 \
        --device cpu
"""

import argparse
import os
import tempfile
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import torch

# ── project imports ──
from ..consts import (
    DEFAULT_MODEL, DEFAULT_MAX_LEN, DEFAULT_SEP, CONTIG_SEP
)
from ..inference import (
    embed_sequence,
    load_base_model,
    load_trained_model,
    softmax_predict,
)
from ..utils import get_sample_id


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def run_query_pipeline_eval(
    fasta_path: str,
    model_path: str,
    base_model_name: str,
    head_model: str,
    device: str,
    max_length: int,
) -> pd.DataFrame:
    """Run the query pipeline in eval mode on a FASTA file and return a DataFrame
    keyed by record.id with columns: pred_serotype, is_cbl, serotype_logits, embedding.

    Uses the same ``inference.embed_sequence`` path as process_trihead_query.py
    with ``inference_mode="eval"`` (single window at position 0).
    """
    # Load both models via the shared inference module — identical to
    # process_trihead_query.main()
    base_bundle = load_base_model(model_name=base_model_name, device=device)
    head_bundle = load_trained_model(model_path, device=device, head_model=head_model)
    logistic_model = head_bundle.model
    idx_to_serotype = head_bundle.idx_to_serotype

    rows = {}
    for record in tqdm(SeqIO.parse(fasta_path, "fasta"), desc="Query-pipeline eval"):
        # Canonical embedding — single source of truth
        cbl_list, sero_list, _geno_list, z_list = embed_sequence(
            base_bundle=base_bundle,
            head_model=logistic_model,
            sequence=str(record.seq),
            device=device,
            max_length=max_length,
            inference_mode="eval",          # <-- single window at pos 0
        )

        sel_cbl = np.asarray(cbl_list[0]).squeeze()
        sel_sero = np.asarray(sero_list[0]).squeeze()
        sel_z = np.asarray(z_list[0])

        cbl_probs = torch.softmax(torch.tensor(sel_cbl, dtype=torch.float32), dim=-1).numpy()
        is_cbl = bool(cbl_probs[1] > 0.5)

        pred_serotype, _, _ = softmax_predict(sel_sero, idx_to_serotype)

        rows[record.id] = {
            "query_pred_serotype": pred_serotype,
            "query_is_cbl": is_cbl,
            "query_serotype_logits": sel_sero,
            "query_embedding": sel_z,
        }

    return pd.DataFrame.from_dict(rows, orient="index")


def run_npz_eval_path(
    npz_path: str,
    model_path: str,
    labels_path: str,
    head_model: str,
    device: str,
    record_ids: set,
) -> pd.DataFrame:
    """Load pre-computed .npz embeddings (z vectors) and run classifier heads directly,
    matching the eval_serotype_classifier.py path. Return DataFrame keyed by record_id."""

    head_bundle = load_trained_model(model_path, device=device, head_model=head_model)
    model = head_bundle.model
    idx_to_serotype = head_bundle.idx_to_serotype

    X = np.load(npz_path, allow_pickle=True)
    labels_df = pd.read_csv(labels_path, index_col=0, sep="\t" if labels_path.endswith(".tsv") else ",")
    labels_df["sample_key"] = (
        labels_df["Is_capsule"].map(lambda x: "cbl" if x else "non-cbl")
        + DEFAULT_SEP
        + get_sample_id(labels_df)
    )
    # Build public_id → sample_key map
    labels_df["public_id"] = labels_df.index

    rows = {}
    for _, row in tqdm(labels_df.iterrows(), desc="NPZ eval path", total=len(labels_df)):
        pid = row["public_id"]
        if pid not in record_ids:
            continue
        key = row["sample_key"]
        if key not in X:
            continue
        z = X[key]
        zt = torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            cbl_logits = model.cbl_classifier(zt)
            serotype_logits = model.serotype_classifier(zt)
        cbl_probs = torch.softmax(cbl_logits, dim=1)[:, 1].cpu().numpy()
        sero_logits_np = serotype_logits.squeeze(0).cpu().numpy()
        pred_serotype, _, _ = softmax_predict(sero_logits_np, idx_to_serotype)

        rows[pid] = {
            "npz_pred_serotype": pred_serotype,
            "npz_is_cbl": bool(cbl_probs[0] > 0.5),
            "npz_serotype_logits": sero_logits_np,
            "npz_embedding": z,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def sample_fasta(fasta_path, labels_path, n_per_serotype, seed=42):
    """Sample n_per_serotype records per serotype from the FASTA, write to a temp file.
    Returns (tmp_fasta_path, set-of-record-ids-sampled)."""
    labels_df = pd.read_csv(labels_path, index_col=0, sep="\t" if labels_path.endswith(".tsv") else ",")
    labels_df = labels_df[labels_df["Serotype"] != "Non-typeable"]
    labels_df = labels_df[labels_df["Is_capsule"].astype(bool)]
    labels_df["fasta_key"] = labels_df.index + CONTIG_SEP + labels_df["Contig_ID"].astype(str)
    rng = random.Random(seed)
    sampled_ids = set()
    for sero, grp in labels_df.groupby("Serotype"):
        ids = grp["fasta_key"].tolist()
        n = min(n_per_serotype, len(ids))
        sampled_ids.update(rng.sample(ids, n))

    print(f"Sampled {len(sampled_ids)} records across {labels_df['Serotype'].nunique()} serotypes.")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".fasta", mode="w")
    count = 0
    for record in SeqIO.parse(fasta_path, "fasta"):
        # record.id is usually public_id__contig format
        pub_id = record.id.split("__")[0]
        if pub_id in sampled_ids:
            SeqIO.write(record, tmp, "fasta")
            count += 1
    tmp.close()
    print(f"Wrote {count} records to {tmp.name}")
    return tmp.name, sampled_ids


def main():
    parser = argparse.ArgumentParser(description="Round-trip sanity check: query pipeline vs npz eval path")
    parser.add_argument("--fasta", required=True, help="Original training FASTA")
    parser.add_argument("--labels", required=True, help="Final metadata TSV")
    parser.add_argument("--model", required=True, help="Trained model .pth")
    parser.add_argument("--inference_npz", required=True, help="Inference .npz from infer_trihead_transformer")
    parser.add_argument("--output_dir", required=True, help="Directory for output report")
    parser.add_argument("--samples_per_serotype", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--base_model", default=DEFAULT_MODEL)
    parser.add_argument("--head_model", default="transformer_trihead_lr")
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LEN)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Sample FASTA records
    print("=== Step 1: Sampling FASTA records ===")
    tmp_fasta, sampled_ids = sample_fasta(args.fasta, args.labels, args.samples_per_serotype)

    # 2) Run query pipeline in eval mode
    print("\n=== Step 2: Running query pipeline (eval mode) ===")
    query_df = run_query_pipeline_eval(
        fasta_path=tmp_fasta,
        model_path=args.model,
        base_model_name=args.base_model,
        head_model=args.head_model,
        device=args.device,
        max_length=args.max_length,
    )

    # Map record IDs to public IDs for joining
    query_df.index = query_df.index.map(lambda x: x.split("__")[0])

    # 3) Run NPZ eval path on the same samples
    print("\n=== Step 3: Running NPZ eval path ===")
    npz_df = run_npz_eval_path(
        npz_path=args.inference_npz,
        model_path=args.model,
        labels_path=args.labels,
        head_model=args.head_model,
        device=args.device,
        record_ids=sampled_ids,
    )

    # 4) Join and compare
    print("\n=== Step 4: Comparing results ===")
    common = query_df.index.intersection(npz_df.index)
    print(f"Common samples: {len(common)} (query: {len(query_df)}, npz: {len(npz_df)})")

    if len(common) == 0:
        os.unlink(tmp_fasta)
        raise ValueError("No common samples between query and npz results. Check ID formats.")

    merged = query_df.loc[common].join(npz_df.loc[common])

    # Comparisons
    serotype_match = (merged["query_pred_serotype"] == merged["npz_pred_serotype"]).astype(int)
    cbl_match = (merged["query_is_cbl"] == merged["npz_is_cbl"]).astype(int)

    cosine_sims = []
    for idx in merged.index:
        q_logits = merged.loc[idx, "query_serotype_logits"]
        n_logits = merged.loc[idx, "npz_serotype_logits"]
        cosine_sims.append(cosine_similarity(q_logits, n_logits))

    emb_cosine_sims = []
    for idx in merged.index:
        q_emb = merged.loc[idx, "query_embedding"]
        n_emb = merged.loc[idx, "npz_embedding"]
        emb_cosine_sims.append(cosine_similarity(q_emb, n_emb))

    merged["serotype_match"] = serotype_match.values
    merged["cbl_match"] = cbl_match.values
    merged["logit_cosine_sim"] = cosine_sims
    merged["embedding_cosine_sim"] = emb_cosine_sims

    # 5) Report
    n = len(merged)
    sero_rate = serotype_match.mean()
    cbl_rate = cbl_match.mean()
    logit_mean = np.mean(cosine_sims)
    emb_mean = np.mean(emb_cosine_sims)

    report_lines = [
        "=" * 60,
        "ROUND-TRIP SANITY CHECK REPORT",
        "=" * 60,
        f"Samples compared:          {n}",
        f"Serotype agreement rate:   {sero_rate:.4f} ({int(serotype_match.sum())}/{n})",
        f"CBL agreement rate:        {cbl_rate:.4f} ({int(cbl_match.sum())}/{n})",
        f"Logit cosine similarity:   {logit_mean:.6f} (mean), {np.min(cosine_sims):.6f} (min)",
        f"Embedding cosine sim:      {emb_mean:.6f} (mean), {np.min(emb_cosine_sims):.6f} (min)",
        "",
    ]

    if sero_rate >= 0.95:
        report_lines.append("PASS: Serotype agreement >= 95%")
    else:
        report_lines.append("FAIL: Serotype agreement < 95%")

    mismatches = merged[merged["serotype_match"] == 0]
    if len(mismatches) > 0:
        report_lines.append(f"\n--- Mismatched samples ({len(mismatches)}) ---")
        for idx in mismatches.index:
            row = mismatches.loc[idx]
            report_lines.append(
                f"  {idx}: query={row['query_pred_serotype']}, npz={row['npz_pred_serotype']}, "
                f"logit_cos={row['logit_cosine_sim']:.4f}, emb_cos={row['embedding_cosine_sim']:.4f}"
            )

    report_text = "\n".join(report_lines)
    print(report_text)

    report_path = os.path.join(args.output_dir, "roundtrip_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    # Save detailed CSV (drop array columns)
    detail_df = merged.drop(columns=[
        "query_serotype_logits", "npz_serotype_logits",
        "query_embedding", "npz_embedding"
    ])
    detail_df.to_csv(os.path.join(args.output_dir, "roundtrip_details.csv"))

    os.unlink(tmp_fasta)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
