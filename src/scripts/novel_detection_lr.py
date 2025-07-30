import argparse
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from typing import List, Tuple
from scipy.stats import beta, norm
from sklearn.metrics.pairwise import cosine_distances
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from models import TransformerLRClassifier
from consts import (
    DEFAULT_MIN_SEROGROUP_SIZE, DEFAULT_MODEL, DEFAULT_CHUNK_SIZE,
    DEFAULT_STRIDE_RATIO, DEFAULT_MAX_LEN
)

EPS = 1e-6
THRESH_CPS = 0.9  # TODO clean up and document and verify and what the fuck
THRESH_NONCPS = 0.1
NORM_NONCBL_PPF = 0.95
THRESH_BETA = 0.98


def transformer_embedding(  # TODO batch this 
    tokenizer,
    nt_model,
    contrastive_model: TransformerLRClassifier,
    sequences: List[str],
    device: str = "cuda",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    stride_ratio: float = DEFAULT_STRIDE_RATIO,
    max_length: int = DEFAULT_MAX_LEN,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a list of sequences, chunk and embed them using a Nucleotide Transformer,
    then feed through the trained TransformerContrastiveHead to get final embeddings.
    Returns: np.ndarray of shape (len(sequences), output_dim)
    """
    max_length = tokenizer.model_max_length
    chunk_size = min(chunk_size, max_length)
    stride = int(chunk_size * stride_ratio)

    all_cbl_logits, all_serotype_logits = [], []
    all_embeddings = []
    for seq in sequences:
        # Chunk the sequence
        chunks = [seq[i:i + chunk_size] for i in range(0, len(seq) - chunk_size + 1, stride)]
        if not chunks:
            continue

        inputs = tokenizer(
            chunks,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = nt_model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]  # (L, T, D)
            pooled = last_hidden.mean(dim=1)         # (L, D)

        # Feed through contrastive model
        with torch.no_grad():
            cbl_logits, serotype_logits, embedding = contrastive_model(pooled.unsqueeze(0))  # (1, L, D) -> (1, output_dim)
            embedding = embedding.squeeze(0).cpu().numpy()  # (output_dim,)
        all_cbl_logits.append(cbl_logits.cpu().numpy())
        all_serotype_logits.append(serotype_logits.cpu().numpy())
        all_embeddings.append(embedding)

    return np.stack(all_cbl_logits), np.stack(all_serotype_logits), np.stack(all_embeddings)


def main(args):
    ### TODO introduce params
    nt_model_name = DEFAULT_MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chunk_size = DEFAULT_CHUNK_SIZE
    stride_ratio = DEFAULT_STRIDE_RATIO
    max_length = DEFAULT_MAX_LEN
    
    thresholds = list(map(float, map(str.strip, args.thresholds.split(","))))
    assert len(thresholds) == 4, "Four thresholds are required."
    assert all(0 < t < 1 for t in thresholds), "Thresholds must be between 0 and 1."

    print("Loading the transformer model and contrastive head...")
    tokenizer = AutoTokenizer.from_pretrained(nt_model_name)
    nt_model = AutoModelForMaskedLM.from_pretrained(nt_model_name).to(device)
    
    model_path = os.path.join(args.output_dir, "transformer_model.pth")
    model_save_dict = torch.load(model_path, map_location=device)
    model_config = model_save_dict['model_config']
    
    # Initialize model with saved configuration
    contrastive_model = TransformerLRClassifier(
        input_dim=model_config['input_dim'],
        num_classes=model_config['num_classes'],
        output_dim=model_config['output_dim'],
        nhead=model_config['nhead'],
        num_layers=model_config['num_layers']
    ).to(device)
    contrastive_model.load_state_dict(model_save_dict['model_state_dict'])
    contrastive_model.eval()

    results = []
    print("Processing queries...")
    total = sum(1 for _ in SeqIO.parse(args.query, "fasta"))
    for record in tqdm(SeqIO.parse(args.query, "fasta"), total=total):
        print(f"Processing query: {record.id}...")
        print("\t- Embedding the query sequence...")
        query_cbl_logits, query_serotype_logits, query_embedding = transformer_embedding(
            tokenizer=tokenizer,
            nt_model=nt_model,
            contrastive_model=contrastive_model,
            sequences=[str(record.seq)],
            device=device,
            chunk_size=chunk_size,
            stride_ratio=stride_ratio,
            max_length=max_length,
        )
        
        query_cbl_logits = query_cbl_logits[0]  # Shape: (num_chunks,)
        query_serotype_logits = query_serotype_logits[0]  # Shape: (num_chunks, num_serotypes)
        
        cbl_mean = np.mean(query_cbl_logits)
        cbl_std = np.std(query_cbl_logits)
        cbl_min = np.min(query_cbl_logits)
        cbl_max = np.max(query_cbl_logits)
        
        serotype_overall_mean = np.mean(query_serotype_logits)
        serotype_overall_std = np.std(query_serotype_logits)
        
        cbl_predictions = torch.sigmoid(torch.tensor(query_cbl_logits)).numpy()
        serotype_predictions = torch.softmax(torch.tensor(query_serotype_logits), dim=-1).numpy()
        cbl_pred_labels = (cbl_predictions > 0.5).astype(int)
        
        serotype_pred_labels = np.argmax(serotype_predictions, axis=-1)
        
        print(f"\t- CBL Logits - Mean: {cbl_mean:.4f}, Std: {cbl_std:.4f}, Min: {cbl_min:.4f}, Max: {cbl_max:.4f}")
        print(f"\t- Serotype Logits - Overall Mean: {serotype_overall_mean:.4f}, Overall Std: {serotype_overall_std:.4f}")
        print(f"\t- CBL Predictions - Mean: {np.mean(cbl_predictions):.4f}, Positive chunks: {np.sum(cbl_pred_labels)}/{len(cbl_pred_labels)}")
        print(f"\t- Serotype Predictions - Most frequent: {np.bincount(serotype_pred_labels).argmax()}, Confidence: {np.max(np.mean(serotype_predictions, axis=0)):.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description="Novel detection script.")
    parser.add_argument("--query", type=str, required=True, help="Path to the query FASTA file.")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to the embeddings file.")
    parser.add_argument("--labels", type=str, required=True, help="Path to the labels file.")  # TODO should be a subset
    parser.add_argument("--distances", type=str, required=True, help="Path to the distances file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output files.")
    parser.add_argument("--distributions", type=str, default="distributions_params.json", help="Path to the distributions parameters file.")
    parser.add_argument("--min_serogroup_size", type=int, default=DEFAULT_MIN_SEROGROUP_SIZE,
        help="Minimum number of samples in a serogroup to be considered for novelty detection."
    )
    parser.add_argument("--thresholds", type=str, default=f"{THRESH_CPS},{THRESH_NONCPS},{NORM_NONCBL_PPF},{THRESH_BETA}",
        help="Comma-separated thresholds for novelty detection."  # TODO: explain the thresholds.
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
