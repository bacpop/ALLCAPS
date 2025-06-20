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


DEFAULT_MISSING_LABEL = "Non-typeable"
DEFAULT_NONCBL_LABEL = "NON-CBL"
DEFAULT_SEP = "|"
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
    missing_label = args.model_params.get("missing_label", DEFAULT_MISSING_LABEL)
    noncbl_label = args.model_params.get("non_cbl_label", DEFAULT_NONCBL_LABEL)
    sep = args.model_params.get("sep", DEFAULT_SEP)
    
    device = torch.device(args.device)
    
    print(f"Loading embeddings from: {args.embeddings_dir}")
    labels = pd.read_csv(args.labels, sep="\t", index_col=0)
    labels['Serotype'] = labels['Serotype'].fillna(missing_label)

    is_duplicate = labels.duplicated()
    if is_duplicate.any():
        print("Dropping duplicate label rows...")
        labels = labels[~is_duplicate]
    if args.skip_labels:
        indices_to_skip = labels['Serotype'].isin(args.skip_labels)
        print(f"Skipping labels: {args.skip_labels}, {indices_to_skip.sum()} rows will be skipped.")
        labels = labels[~indices_to_skip]
    
    known_indices = labels['Serotype'] != missing_label
    labels = labels[known_indices]

    noncbl_subdir = os.path.join(args.embeddings_dir, "non-cbl")
    if os.path.exists(noncbl_subdir):
        print("Found non-cbl embeddings, adding NON-CBL label and embeddings.")
        non_cbl_embeddings = [f for f in os.listdir(noncbl_subdir) if f.endswith('.npy')]
        non_cbl_embeddings = [f.replace('.npy', '') for f in non_cbl_embeddings]
        
        non_cbl_labels = pd.DataFrame({
            'Serotype': [noncbl_label] * len(non_cbl_embeddings),
        }, index=non_cbl_embeddings)
        labels = pd.concat([labels, non_cbl_labels], axis=0)

    sequences = labels.index.tolist()
    serotype_labels = labels["Serotype"].tolist()
    capsule_labels = (labels["Serotype"] != noncbl_label).astype(int).tolist()

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
            is_capsule = batch["is_capsule"].to(device)

            _, z = model(inputs)  # contrastive embeddings
            z = z.cpu().numpy()

            for emb, sid, cbl in zip(z, sample_ids, is_capsule):
                pref = "cbl" if cbl else "non-cbl"
                all_embeddings[f"{pref}{sep}{sid}"] = emb

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
    parser.add_argument("--skip_labels", type=str, default="",
                        help="Comma-separated list of labels to skip in training.")
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

    try:
        args.skip_labels = [label.strip() for label in args.skip_labels.split(",") if label.strip()]
    except ValueError:
        print("Error parsing skip_labels. It should be a comma-separated list of labels. Proceeding with no skips.")
        args.skip_labels = []

    return args


if __name__ == "__main__":
    main(parse_args())
