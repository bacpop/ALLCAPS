#!/usr/bin/env python

import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim

from models import ContrastiveHead
from utils import map_serotype_to_group

EPS = 1e-9
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_TEMPERATURE = 0.07
DEFAULT_KFOLDS = 5
MISSING_LABEL = "Non-typeable"  # TODO: Make this a parameter
LABEL_COLUMN = "serotype"  # TODO: Do sth about it


def supervised_contrastive_loss(z, labels, temperature):
    """
    Supervised Contrastive Loss:
      - For each anchor i, all samples j with the same label are positives.
      - Different labels => negatives.
      - i != j (exclude diagonal).
    """
    device, N = z.device, z.shape[0]
    z = nn.functional.normalize(z, dim=1)  # Normalize embeddings for stable similarity
    logits = z @ z.t() / temperature  # shape (N, N)

    # positives_mask = torch.tensor([[(labels[i] == labels[j]) and (i != j) for j in range(N)] for i in range(N)], device=device)
    positives_mask = torch.zeros((N, N), dtype=torch.bool, device=device)
    for i in range(N):
        for j in range(N):
            if i != j and labels[i] == labels[j]:
                positives_mask[i, j] = True

    diag_mask = torch.eye(N, dtype=torch.bool, device=device)  # Exclude diagonal from denominator

    exp_logits = torch.exp(logits)
    pos_exp = exp_logits * positives_mask
    pos_sum = pos_exp.sum(dim=1)  # (N,)

    den_exp = exp_logits * ~diag_mask
    den_sum = den_exp.sum(dim=1)

    loss_terms = -torch.log((pos_sum + EPS) / (den_sum + EPS))
    loss = loss_terms.mean()
    return loss


def train_one_epoch(model, X_train, labels_train, optimizer, batch_size, temperature):
    """
    Trains the model for one epoch using supervised contrastive loss.
    Returns the average training loss for this epoch.
    """
    model.train()
    device = next(model.parameters()).device
    n = X_train.shape[0]

    # Shuffle
    perm = torch.randperm(n, device=device)
    X_shuffled = X_train[perm]
    labels_shuffled = [labels_train[i.item()] for i in perm]

    total_loss = 0.0
    num_batches = n // batch_size

    for b in range(num_batches):
        start = b * batch_size
        end = (b + 1) * batch_size

        batch_data = X_shuffled[start:end]  # (batch_size, D)
        batch_labels = labels_shuffled[start:end]

        z = model(batch_data)
        loss = supervised_contrastive_loss(z, batch_labels, temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (num_batches + EPS)
    return avg_loss


@torch.no_grad()
def evaluate_loss(model, X_val, labels_val, batch_size, temperature):
    """
    Computes the supervised contrastive loss on the validation set,
    returning the average loss as a 'validation metric'.
    """
    model.eval()
    device = next(model.parameters()).device
    n = X_val.shape[0]

    perm = torch.arange(n, device=device)
    X_reorder = X_val[perm]
    labels_reorder = [labels_val[i.item()] for i in perm]

    total_loss = 0.0
    num_batches = min(1, n // batch_size)

    for b in range(num_batches):
        start = b * batch_size
        end = (b + 1) * batch_size

        batch_data = X_reorder[start:end]
        batch_labels = labels_reorder[start:end]
        if len(batch_data) == 0: break

        z = model(batch_data)
        loss = supervised_contrastive_loss(z, batch_labels, temperature=temperature)
        total_loss += loss.item()

    avg_loss = total_loss / (num_batches + EPS)
    return avg_loss


def main(args):
    device = args.device
    k_folds = args.model_params.get("k_folds", DEFAULT_KFOLDS)
    temperature = args.model_params.get("temperature", DEFAULT_TEMPERATURE)
    random_state = args.model_params.get("random_state", 42)

    print("Loading data...")
    X = np.load(args.embeddings)  # shape (N, D)
    labels = pd.read_csv(args.labels, sep="\t", index_col=0)
    assert X.shape[0] == len(labels), "Number of embeddings and labels do not match."
    labels['Serotype'] = labels[LABEL_COLUMN].fillna(MISSING_LABEL)
    
    # Sorry about this, Sam
    if True:  # TODO Will deal with subclasses later. Maybe another head?
        labels['Serotype'] = labels['Serotype'].apply(map_serotype_to_group)

    indices = labels["Serotype"] != MISSING_LABEL if args.labeled_only else np.ones(len(labels), dtype=bool)
    X_known = X[indices]
    labels_known = labels['Serotype'][indices].values.tolist()
    print(f"Total samples: {X.shape[0]}, of which using: {X_known.shape[0]}")

    X_torch = torch.from_numpy(X_known).float().to(device)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    fold_metrics, fold_idx = [], 0
    for train_idx, val_idx in tqdm(skf.split(X_torch.cpu().numpy(), labels_known),
                                   desc="Cross-validation", leave=True, position=0):
        fold_idx += 1

        X_train = X_torch[train_idx]
        y_train = [labels_known[i] for i in train_idx]
        X_val = X_torch[val_idx]
        y_val = [labels_known[i] for i in val_idx]

        input_dim = X_train.shape[1]
        model = ContrastiveHead(input_dim=input_dim, output_dim=128).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        for epoch in tqdm(range(args.epochs), desc=f"Fold {fold_idx}/{k_folds}", leave=False, position=1):
            train_loss = train_one_epoch(model, X_train, y_train, optimizer, args.batch_size, temperature)
            val_loss = evaluate_loss(model, X_val, y_val, args.batch_size, temperature)
            tqdm.write(f"Epoch {epoch + 1}/{args.epochs}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Final val loss after training
        final_val_loss = evaluate_loss(model, X_val, y_val, args.batch_size, temperature)
        tqdm.write(f"Fold {fold_idx} final val_loss: {final_val_loss:.4f}")
        fold_metrics.append(final_val_loss)

    # Summarize cross-validation performance
    fold_metrics = np.array(fold_metrics)
    mean_metric, std_metric = fold_metrics.mean(), fold_metrics.std()
    print(f"\nCross-validation results ({k_folds}-fold):")
    print(f"Mean val_loss = {mean_metric:.4f}, Std = {std_metric:.4f}")

    print("\nRetraining on entire known-labeled dataset for final model...")
    model_final = ContrastiveHead(input_dim=X_torch.shape[1], output_dim=128).to(device)
    optimizer_final = optim.Adam(model_final.parameters(), lr=args.lr)

    is_cuda = next(model_final.parameters()).is_cuda
    print(f"Contrastive head is using CUDA: {is_cuda}")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model_final, X_torch, labels_known, optimizer_final, args.batch_size, temperature)
        print(f"Retrain Epoch {epoch + 1}/{args.epochs}, train_loss={train_loss:.4f}")

    torch.save(model_final.state_dict(), args.output)
    print(f"Saved final model to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--model-params", type=str, default="{}",
                        help="JSON string of model parameters.")
    parser.add_argument("--labeled-only", action="store_true",
                        help="Use only labeled data for training.")
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
