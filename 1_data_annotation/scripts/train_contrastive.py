#!/usr/bin/env python

import os
import json
import wandb
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim

from models import TransformerContrastiveHead
from utils import map_serotype_to_group, supervised_contrastive_loss, hierarchical_contrastive_loss

EPS = 1e-9
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_TEMPERATURE = 0.07
DEFAULT_KFOLDS = 5
DEFAULT_WEIGHT_FINE = 1.0
DEFAULT_WEIGHT_COARSE = 0.6
DEFAULT_NUM_LAYERS = 2
DEFAULT_NHEAD = 4
DEFAULT_DIM_FEEDFORWARD = 512

MISSING_LABEL = "Non-typeable"  # TODO: Make this a parameter
LABEL_COLUMN = "Serotype"  # TODO: Do sth about it
PROJECT_NAME = "contrastive-training"


def train_one_epoch(model, loss_fn, X_train, labels_train, optimizer, batch_size, temperature):
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
        loss = loss_fn(z, batch_labels, temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (num_batches + EPS)
    return avg_loss


@torch.no_grad()
def evaluate_loss(model, loss_fn, X_val, labels_val, batch_size, temperature):
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
        loss = loss_fn(z, batch_labels, temperature=temperature)
        total_loss += loss.item()

    avg_loss = total_loss / (num_batches + EPS)
    return avg_loss


def main(args):
    device = args.device
    random_state = args.model_params.get("random_state", 42)
    k_folds = args.model_params.get("k_folds", DEFAULT_KFOLDS)
    temperature = args.model_params.get("temperature", DEFAULT_TEMPERATURE)
    weight_fine = args.model_params.get("weight_fine", DEFAULT_WEIGHT_FINE)
    weight_coarse = args.model_params.get("weight_coarse", DEFAULT_WEIGHT_COARSE)
    num_layers = args.model_params.get("num_layers", DEFAULT_NUM_LAYERS)
    nhead = args.model_params.get("nhead", DEFAULT_NHEAD)
    dim_feedforward = args.model_params.get("dim_feedforward", DEFAULT_DIM_FEEDFORWARD)
    wandb.config.update({
        "random_state": random_state,
        "k_folds": k_folds,
        "temperature": temperature,
        "weight_fine": weight_fine,
        "weight_coarse": weight_coarse,
        "num_layers": num_layers,
        "nhead": nhead,
        "dim_feedforward": dim_feedforward,
    })

    print("Loading data...")
    X = np.load(args.embeddings)  # shape (N, D)
    labels = pd.read_csv(args.labels, sep="\t", index_col=0)
    assert X.shape[0] == len(labels), "Number of embeddings and labels do not match."
    labels['Serotype'] = labels[LABEL_COLUMN].fillna(MISSING_LABEL)

    indices = labels["Serotype"] != MISSING_LABEL if args.labeled_only else np.ones(len(labels), dtype=bool)
    if args.skip_labels:
        print(f"Skipping labels: {args.skip_labels}")
        indices &= ~labels['Serotype'].isin(args.skip_labels)
    X_known = X[indices]
    fine_labels = labels['Serotype'][indices].values.tolist()

    loss_function = supervised_contrastive_loss
    labels_known = fine_labels
    if args.hierarchical_loss:
        print("Using hierarchical contrastive loss with weights:", weight_fine, weight_coarse)
        loss_function = partial(hierarchical_contrastive_loss, weight_fine=weight_fine, weight_coarse=weight_coarse)
        coarse_labels = labels['Serotype'][indices].apply(map_serotype_to_group).values.tolist()
        labels_known = list(zip(coarse_labels, fine_labels))

    print(f"Total samples: {X.shape[0]}, of which using: {X_known.shape[0]}")

    X_torch = torch.from_numpy(X_known).float().to(device)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    fold_metrics, fold_idx = [], 0
    # TODO Should I base the stratification on coarse labels or fine labels? Proceeding with fine for now
    for train_idx, val_idx in tqdm(skf.split(X_torch.cpu().numpy(), fine_labels),
                                   desc="Cross-validation", leave=True, position=0):
        fold_idx += 1

        X_train = X_torch[train_idx]
        y_train = [labels_known[i] for i in train_idx]
        X_val = X_torch[val_idx]
        y_val = [labels_known[i] for i in val_idx]

        input_dim = X_train.shape[1]
        model = TransformerContrastiveHead(input_dim, nhead, num_layers, dim_feedforward).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        for epoch in tqdm(range(args.epochs), desc=f"Fold {fold_idx}/{k_folds}", leave=False, position=1):
            train_loss = train_one_epoch(model, loss_function, X_train, y_train, optimizer, args.batch_size,
                                         temperature)
            val_loss = evaluate_loss(model, loss_function, X_val, y_val, args.batch_size, temperature)
            tqdm.write(f"Epoch {epoch + 1}/{args.epochs}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch + 1
            })

        # Final val loss after training
        final_val_loss = evaluate_loss(model, loss_function, X_val, y_val, args.batch_size, temperature)
        tqdm.write(f"Fold {fold_idx} final val_loss: {final_val_loss:.4f}")
        fold_metrics.append(final_val_loss)

    # Summarize cross-validation performance
    fold_metrics = np.array(fold_metrics)
    mean_metric, std_metric = fold_metrics.mean(), fold_metrics.std()
    print(f"\nCross-validation results ({k_folds}-fold):")
    print(f"Mean val_loss = {mean_metric:.4f}, Std = {std_metric:.4f}")
    wandb.summary["mean_val_loss"] = mean_metric
    wandb.summary["std_val_loss"] = std_metric

    print("\nRetraining on entire known-labeled dataset for final model...")
    model_final = TransformerContrastiveHead(X_torch.shape[1], nhead, num_layers, dim_feedforward).to(device)
    optimizer_final = optim.Adam(model_final.parameters(), lr=args.lr)

    is_cuda = next(model_final.parameters()).is_cuda
    print(f"Contrastive head is using CUDA: {is_cuda}")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model_final, loss_function, X_torch, labels_known, optimizer_final, args.batch_size, temperature)
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
    parser.add_argument("--hierarchical-loss", action="store_true",
                        help="Use weighted (coarse, fine) labels for training.")
    # a new argument that receives a list of strings to skip labels in training
    parser.add_argument("--skip-labels", type=str, default="",
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
    args = parse_args()
    run_id = os.environ.get("SLURM_JOB_ID", os.urandom(4).hex())
    mode = "offline" if os.environ.get("WANDB_MODE") == "offline" else "online"
    wandb.init(
        project=PROJECT_NAME,
        config=args,
        mode=mode
    )
    wandb.run.name = f"{PROJECT_NAME}-{run_id}"
    main(args)
