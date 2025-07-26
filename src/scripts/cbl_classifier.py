import json
import argparse
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from models import TransformerContrastiveHead
from consts import (
    DEFAULT_NUM_LAYERS, DEFAULT_NHEAD, DEFAULT_OUTPUT_DIM,
    DEFAULT_EMBEDDING_DIM, DEFAULT_SEP, DEFAULT_BATCH_SIZE,
)


def main(args):
    num_layers = args.model_params.get("num_layers", DEFAULT_NUM_LAYERS)
    nhead = args.model_params.get("nhead", DEFAULT_NHEAD)
    output_dim = args.model_params.get("output_dim", DEFAULT_OUTPUT_DIM)
    embedding_dim = args.model_params.get("embedding_dim", DEFAULT_EMBEDDING_DIM)
    sep = args.model_params.get("sep", DEFAULT_SEP)
    
    print(f"Loading embeddings from: {args.embeddings}")
    X = np.load(args.embeddings)  # shape: (N, L, D)
    labels = list(map(lambda sid: sid.split(sep), X.keys()))  # assuming keys are in the format "capsule_label|public_name"
    labels = pd.DataFrame(labels, columns=["Capsule_label", "Public_name"])
    labels['Capsule_label'] = labels['Capsule_label'].map(lambda x: 1 if x == "cbl" else 0).astype(int)  # TODO clean up this shit
    
    X = np.stack([X[k] for k in X.keys()])
    y = labels['Capsule_label'].values

    print(f"Loading model from: {args.model}")
    device = torch.device(args.device)
    model = TransformerContrastiveHead(
        input_dim=embedding_dim,
        output_dim=output_dim,
        nhead=nhead,
        num_layers=num_layers
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    print("Running inference...")
    all_logits = []
    with torch.no_grad():
        for i in tqdm(range(0, len(X), args.batch_size)):
            batch = torch.tensor(X[i:i+args.batch_size], dtype=torch.float32, device=device)
            logits = model.classifier(batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_logits.append(preds)
    y_pred = np.concatenate(all_logits)
    clf_report = classification_report(y, y_pred, target_names=["Non-CBL", "CBL"], labels=[0, 1])
    f1 = f1_score(y, y_pred, average='weighted')
    
    print("Classification report:")
    print(clf_report)
    print("F1 score (weighted):", f1)
    with open(args.output, 'w') as f:
        f.write("Classification report:\n")
        f.write(clf_report)
        f.write("\nF1 score (weighted): {}\n".format(f1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, help="Path to .npy embeddings file (N, L, D)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="cbl_classifier_output.txt", help="Output file for classification report")
    parser.add_argument("--model", required=True)
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

    main(args)
