import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from models import TransformerLRClassifier
from consts import (
    DEFAULT_MIN_SEROGROUP_SIZE, DEFAULT_MODEL, DEFAULT_CHUNK_SIZE,
    DEFAULT_STRIDE_RATIO, DEFAULT_MAX_LEN
)
from utils import chunk_sequence, embed_chunks


EPS = 1e-6
THRESH_CPS = 0.5  # TODO clean up and document and verify and what the fuck
THRESH_NONCPS = 0.1
NORM_NONCBL_PPF = 0.95
THRESH_BETA = 0.98


def energy_score(logits, temperature=1.0) -> float:
    energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
    return energy.item()


def transformer_embedding(  # TODO batch this 
    tokenizer,
    nt_model,
    logistic_model: TransformerLRClassifier,
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
        chunks = chunk_sequence(seq, chunk_size, stride)
        if not chunks:
            print(f"Skipping sequence due to no valid chunks: {seq[:30]}...")
            continue
        pooled = embed_chunks(chunks, tokenizer, nt_model, device, max_length)  # shape (L, D)    

        # Feed through logistic transformer model
        with torch.no_grad():
            cbl_logits, serotype_logits, embedding = logistic_model(pooled.unsqueeze(0))  # (1, L, D) -> (1, output_dim)
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
    idx_to_serotype = {v: k for k, v in model_save_dict['serotype_to_idx'].items()}
    
    # Initialize model with saved configuration
    logistic_model = TransformerLRClassifier(
        input_dim=model_config['input_dim'],
        num_classes=model_config['num_classes'],
        output_dim=model_config['output_dim'],
        nhead=model_config['nhead'],
        num_layers=model_config['num_layers']
    ).to(device)
    logistic_model.load_state_dict(model_save_dict['model_state_dict'])
    logistic_model.eval()

    results = dict()
    print("Processing queries...")
    total = sum(1 for _ in SeqIO.parse(args.query, "fasta"))
    for record in tqdm(SeqIO.parse(args.query, "fasta"), total=total):
        print(f"Processing query: {record.id}...")
        query_cbl_logits, query_serotype_logits, query_embedding = transformer_embedding(
            tokenizer=tokenizer,
            nt_model=nt_model,
            logistic_model=logistic_model,
            sequences=[str(record.seq)],
            device=device,
            chunk_size=chunk_size,
            stride_ratio=stride_ratio,
            max_length=max_length,
        )
        
        query_cbl_logits = query_cbl_logits[0][0]  # (1, 2) -> (2,)
        query_serotype_logits = query_serotype_logits[0]
        cbl_predictions = torch.sigmoid(torch.tensor(query_cbl_logits)).numpy()
        
        results[record.id] = {
            "cbl_logits": query_cbl_logits,
            "serotype_logits": query_serotype_logits,
            "embedding": query_embedding,
            "is_cbl": cbl_predictions[1] > thresholds[0],
        }
    
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df["pred_argmax"] = results_df["serotype_logits"].apply(
        lambda x: idx_to_serotype[np.argmax(torch.softmax(torch.tensor(x), dim=-1).numpy())]
    )
    results_df["pred_cbl"] = results_df["cbl_logits"].apply(lambda x: "cbl" if x[1] > thresholds[0] else "non-cbl")
    results_df["logits_energy"] = results_df["serotype_logits"].apply(lambda x: energy_score(torch.tensor(x), temperature=1.0))

    results_df.to_csv(os.path.join(args.output_dir, "query_results.csv"), index=False)


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
