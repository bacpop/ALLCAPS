#!/usr/bin/env python

import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from models import TransformerContrastiveHead, ContrastiveChunkedDataset
from utils import collate_fn


MISSING_LABEL = "Non-typeable"
NONCBL_LABEL = "NON-CBL"
DEFAULT_OUTPUT_DIM = 128
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_LAYERS = 1
DEFAULT_NHEAD = 4
DEFAULT_EMBEDDING_DIM = 2560  # Nucleotide Transformer output TODO


def main(args):
    num_layers = args.model_params.get("num_layers", DEFAULT_NUM_LAYERS)
    nhead = args.model_params.get("nhead", DEFAULT_NHEAD)
    output_dim = args.model_params.get("output_dim", DEFAULT_OUTPUT_DIM)
    embedding_dim = args.model_params.get("embedding_dim", DEFAULT_EMBEDDING_DIM)

    device = torch.device(args.device)
    
    print(f"Loading embeddings from: {args.embeddings_dir}")
    labels = pd.read_csv(args.labels, sep="\t", index_col=0)
    labels['Serotype'] = labels['Serotype'].fillna(MISSING_LABEL)

    is_duplicate = labels.duplicated()
    if is_duplicate.any():
        print("Dropping duplicate label rows...")
        labels = labels[~is_duplicate]

    known_indices = labels['Serotype'] != MISSING_LABEL
    labels = labels[known_indices]

    sequences = labels.index.tolist()
    serotype_labels = labels["Serotype"].tolist()
    capsule_labels = (labels["Serotype"] != NONCBL_LABEL).astype(int).tolist()

    dataset = ContrastiveChunkedDataset(args.embeddings_dir, sequences, serotype_labels, capsule_labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    print(f"Using model: {args.model}")
    model = TransformerContrastiveHead(
        input_dim=embedding_dim,
        output_dim=output_dim,
        nhead=nhead,
        num_layers=num_layers
    ).to(device)

    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    print(f"Model loaded with input dimension: {embedding_dim}, output dimension: {output_dim}")
    print(f"Number of samples: {len(dataset)}")
    all_embeddings = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            inputs = batch["embedding"].to(device)
            sample_ids = batch["sample_id"]

            _, z = model(inputs)  # contrastive embeddings
            z = z.cpu().numpy()

            for emb, sid in zip(z, sample_ids):
                all_embeddings[sid] = emb

    np.savez_compressed(args.output, **all_embeddings)
    print(f"Embeddings saved to: {args.output}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--model_params", type=str, default="{}")
    args = parser.parse_args()
    try:
        args.model_params = json.loads(args.model_params)
        if not isinstance(args.model_params, dict):
            print("Model parameters should be a JSON object.")
            args.model_params = {}
    except json.JSONDecodeError:
        print("Error parsing model parameters JSON string.")
        args.model_params = {}
    finally:
        print("Model parameters:", args.model_params)
    
    return args


if __name__ == "__main__":
    main(parse_args())
