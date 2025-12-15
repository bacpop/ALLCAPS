#!/usr/bin/env python

import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from models import TransformerLRClassifier, ContrastiveChunkedDataset
from utils import collate_fn, get_sample_id

from consts import (
    DEFAULT_MISSING_LABEL, DEFAULT_SEP, DEFAULT_BATCH_SIZE, DEFAULT_LABEL_COLUMN
)


def main(args):
    missing_label = args.model_params.get("missing_label", DEFAULT_MISSING_LABEL)
    label_column = args.model_params.get("label_column", DEFAULT_LABEL_COLUMN)
    sep = args.model_params.get("sep", DEFAULT_SEP)
    
    device = torch.device(args.device)

    print(f"Loading labels from: {args.labels}")
    labels = pd.read_csv(args.labels, sep="\t", index_col=0)
    labels['Serotype'] = labels[label_column].fillna(missing_label)

    if args.skip_labels:
        indices_to_skip = labels['Serotype'].isin(args.skip_labels)
        print(f"Skipping labels: {args.skip_labels}, {indices_to_skip.sum()} rows will be skipped.")
        labels = labels[~indices_to_skip]

    if args.labeled_only:
        known_indices = labels['Serotype'] != missing_label
        labels = labels[known_indices]

    sample_ids = get_sample_id(labels).tolist()
    serotype_labels = labels["Serotype"].tolist()
    capsule_labels = labels["Is_capsule"].tolist()

    dataset = ContrastiveChunkedDataset(args.embeddings_dir, sample_ids, serotype_labels, capsule_labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    print(f"Loading model from: {args.model}")
    model_save_dict = torch.load(args.model, map_location=device)
    model_config = model_save_dict['model_config']
    serotype_to_idx = model_save_dict['serotype_to_idx']
    num_serotypes = model_save_dict['num_serotypes']
    
    print(f"Model configuration: {model_config}")
    print(f"Number of serotypes: {num_serotypes}")
    
    model = TransformerLRClassifier(
        input_dim=model_config['input_dim'],
        num_classes=model_config['num_classes'],
        output_dim=model_config['output_dim'],
        nhead=model_config['nhead'],
        num_layers=model_config['num_layers']
    ).to(device)
    model.load_state_dict(model_save_dict['model_state_dict'])
    model.eval()

    print(f"Model loaded with input dimension: {model_config['input_dim']}, output dimension: {model_config['output_dim']}")
    print(f"Number of samples: {len(dataset)}")
    
    all_embeddings = {}
    all_cbl_predictions = {}
    all_serotype_predictions = {}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings and predictions"):
            inputs = batch["embedding"].to(device)
            sample_ids = batch["sample_id"]
            is_capsule = batch["is_capsule"].to(device)

            cbl_logits, serotype_logits, z = model(inputs)
            z_np = z.cpu().numpy()
            cbl_probs = torch.softmax(cbl_logits, dim=1).cpu().numpy()
            serotype_probs = torch.softmax(serotype_logits, dim=1).cpu().numpy()

            for i, (emb, sid, cbl, cbl_prob, sero_prob) in enumerate(zip(z_np, sample_ids, is_capsule, cbl_probs, serotype_probs)):
                pref = "cbl" if cbl else "non-cbl"
                key = f"{pref}{sep}{sid}"
                
                all_embeddings[key] = emb
                all_cbl_predictions[key] = cbl_prob[1]  # Probability of class 1 (capsulated)
                
                if cbl:
                    predicted_idx = np.argmax(sero_prob)
                    idx_to_serotype = {v: k for k, v in serotype_to_idx.items()}
                    predicted_serotype = idx_to_serotype[predicted_idx]
                    all_serotype_predictions[key] = {
                        'predicted_serotype': predicted_serotype,
                        'confidence': sero_prob[predicted_idx],
                        'probabilities': sero_prob  # All serotype probabilities
                    }

    np.savez_compressed(args.output, **all_embeddings)
    print(f"Embeddings saved to: {args.output}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_dir", required=True, 
                        help="Directory containing chunked embeddings in npy format.")
    parser.add_argument("--labels", required=True,
                        help="TSV file with labels indexed by sample ID.")
    parser.add_argument("--model", required=True,
                        help="Path to the saved model file (.pth).")
    parser.add_argument("--output", required=True,
                        help="Output path for the compressed embeddings (.npz).")
    parser.add_argument("--device", default="cpu",
                        help="Device to use for inference (cpu or cuda).")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Batch size for inference.")
    parser.add_argument("--model_params", type=str, default="{}",
                        help="JSON string of model parameters.")
    parser.add_argument("--skip_labels", type=str, default="",
                        help="Comma-separated list of labels to skip in inference.")
    parser.add_argument("--labeled_only", action="store_true",
                        help="Only process samples with known labels.")
    
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
