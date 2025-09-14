import os
import json
import wandb
import argparse
from tqdm import tqdm
from functools import partial

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from models import TransformerContrastiveHead, ContrastiveChunkedDataset
from utils import supervised_contrastive_loss, hierarchical_contrastive_loss, map_serotype_to_group, collate_fn

from consts import (
    RND_STATE, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LR,
    DEFAULT_KFOLDS, DEFAULT_TEMPERATURE, DEFAULT_WEIGHT_FINE,
    DEFAULT_WEIGHT_COARSE, DEFAULT_NUM_LAYERS, DEFAULT_NHEAD,
    DEFAULT_OUTPUT_DIM, DEFAULT_EMBEDDING_DIM, CONTIG_SEP,
    DEFAULT_MISSING_LABEL, DEFAULT_NONCBL_LABEL, DEFAULT_LABEL_COLUMN,
    DEFAULT_EARLY_STOPPING, DEFAULT_CONTRASTIVE_LOSS_RATIO
)

EPS = 1e-9
WANDB_PROJECT_NAME = "contrastive-inference"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dir", required=True, help="Directory containing chunked embeddings in npy format.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--model_params", type=str, default="{}",
                        help="JSON string of model parameters (output_dim, num_layers, nhead, alpha, temperature, etc.)")
    parser.add_argument("--labeled_only", action="store_true")
    parser.add_argument("--skip_labels", type=str, default="",
                        help="Comma-separated list of labels to skip in training.")
    parser.add_argument("--hierarchical_loss", action="store_true",
                        help="Use weighted (coarse, fine) labels for training.")
    parser.add_argument("--early_stopping", type=int, default=DEFAULT_EARLY_STOPPING)
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


def train_one_epoch(model, loader, optimizer, ce_loss_fn, contrastive_loss_fn, alpha, temperature):
    model.train()
    total_loss, ce_loss, contrastive_loss = 0.0, 0.0, 0.0
    for batch in loader:
        capsule_label = batch['is_capsule'].cuda()
        serotype_label = batch['serotype']
        capsule_mask = (capsule_label == 1)

        logits, embeddings = model(batch['embedding'].cuda())

        ce_loss = ce_loss_fn(logits, capsule_label)

        if capsule_mask.sum() > 1:
            contrastive_loss = contrastive_loss_fn(embeddings[capsule_mask], [serotype_label[i] for i in range(len(capsule_label)) if capsule_mask[i]], temperature)
        else:
            contrastive_loss = torch.tensor(0.0, device=ce_loss.device)

        loss = ce_loss + alpha * contrastive_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        ce_loss += ce_loss.item()
        contrastive_loss += contrastive_loss.item()

    wandb.log({
        "epoch_loss": total_loss / len(loader),
        "epoch_ce_loss": ce_loss / len(loader),
        "epoch_contrastive_loss": contrastive_loss / len(loader)
    })
    return total_loss / len(loader)


def evaluate(model, loader, ce_loss_fn, contrastive_loss_fn, alpha, temperature):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            capsule_label = batch['is_capsule'].cuda()
            serotype_label = batch['serotype']
            capsule_mask = (capsule_label == 1)

            logits, embeddings = model(batch['embedding'].cuda())
            ce_loss = ce_loss_fn(logits, capsule_label)

            if capsule_mask.sum() > 1:
                contrastive_loss = contrastive_loss_fn(embeddings[capsule_mask], [serotype_label[i] for i in range(len(capsule_label)) if capsule_mask[i]], temperature)
            else:
                contrastive_loss = torch.tensor(0.0, device=ce_loss.device)

            loss = ce_loss + alpha * contrastive_loss
            total_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            correct += (predicted == capsule_label).sum().item()
            total += capsule_label.size(0)

    accuracy = correct / total if total > 0 else 0.0
    wandb.log({
        "test_loss": total_loss / len(loader),
        "accuracy": accuracy,
    })
    return total_loss / len(loader), accuracy


def main(args):
    device = args.device
    random_state = args.model_params.get("random_state", RND_STATE)
    k_folds = args.model_params.get("k_folds", DEFAULT_KFOLDS)
    temperature = args.model_params.get("temperature", DEFAULT_TEMPERATURE)
    weight_fine = args.model_params.get("weight_fine", DEFAULT_WEIGHT_FINE)
    weight_coarse = args.model_params.get("weight_coarse", DEFAULT_WEIGHT_COARSE)
    num_layers = args.model_params.get("num_layers", DEFAULT_NUM_LAYERS)
    nhead = args.model_params.get("nhead", DEFAULT_NHEAD)
    alpha = args.model_params.get("alpha", DEFAULT_CONTRASTIVE_LOSS_RATIO)
    output_dim = args.model_params.get("output_dim", DEFAULT_OUTPUT_DIM)
    embedding_dim = args.model_params.get("embedding_dim", DEFAULT_EMBEDDING_DIM)

    missing_label = args.model_params.get("missing_label", DEFAULT_MISSING_LABEL)
    label_column = args.model_params.get("label_column", DEFAULT_LABEL_COLUMN)

    wandb.config.update({
        "random_state": random_state,
        "k_folds": k_folds,
        "temperature": temperature,
        "weight_fine": weight_fine,
        "weight_coarse": weight_coarse,
        "num_layers": num_layers,
        "nhead": nhead,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "alpha": alpha,
        "output_dim": output_dim,
    })
    
    print("Loading data...")
    labels = pd.read_csv(args.labels, sep="\t", index_col=0)
    labels['Serotype'] = labels[label_column].fillna(missing_label)

    indices = labels["Serotype"] != missing_label if args.labeled_only else np.ones(len(labels), dtype=bool)
    if args.skip_labels:
        skip_indices = labels['Serotype'].isin(args.skip_labels)
        print(f"Skipping labels: {args.skip_labels} accounting for {skip_indices.sum()} samples.")
        indices &= ~skip_indices

    fine_labels = labels['Serotype'][indices].values.tolist()
    if args.hierarchical_loss:
        print("Using hierarchical contrastive loss with weights:", weight_fine, weight_coarse)
        coarse_labels = labels['Serotype'][indices].apply(map_serotype_to_group).tolist()
        labels_known = list(zip(coarse_labels, fine_labels))
        loss_function = partial(hierarchical_contrastive_loss, weight_fine=weight_fine, weight_coarse=weight_coarse)
    else:
        labels_known = fine_labels
        loss_function = supervised_contrastive_loss

    sample_ids = (labels.index[indices] + CONTIG_SEP + labels["Contig_ID"][indices]).tolist()
    is_capsule = labels["Is_capsule"][indices].tolist()

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(fine_labels)), fine_labels)):  # Dummy X
        print(f"Fold {fold+1} / {k_folds}")

        train_ds = ContrastiveChunkedDataset(args.embedding_dir, np.array(sample_ids)[train_idx],
                                             np.array(labels_known)[train_idx], np.array(is_capsule)[train_idx])
        test_ds = ContrastiveChunkedDataset(args.embedding_dir, np.array(sample_ids)[test_idx],
                                            np.array(labels_known)[test_idx], np.array(is_capsule)[test_idx])

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate_fn)

        model = TransformerContrastiveHead(
            input_dim=embedding_dim,
            output_dim=output_dim,
            nhead=nhead,
            num_layers=num_layers
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        ce_loss_fn = nn.CrossEntropyLoss()

        best_loss, patience_counter = float('inf'), 0

        for epoch in tqdm(range(args.epochs), desc=f"Training Fold {fold+1}"):
            train_one_epoch(model, train_loader, optimizer, ce_loss_fn, loss_function, alpha, temperature)

            test_loss, accuracy = evaluate(model, test_loader, ce_loss_fn, loss_function, alpha, temperature)
            print(f"Fold {fold+1} - Epoch {epoch+1} - Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.early_stopping:
                    print("Early stopping triggered.")
                    break

    # Retrain on all data
    print("Retraining on all data...")
    all_ds = ContrastiveChunkedDataset(args.embedding_dir, sample_ids, labels_known, is_capsule)
    all_loader = DataLoader(all_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model_final = TransformerContrastiveHead(
        input_dim=embedding_dim,
        output_dim=output_dim,
        nhead=nhead,
        num_layers=num_layers
    ).to(device)

    optimizer = torch.optim.AdamW(model_final.parameters(), lr=args.lr)
    ce_loss_fn = nn.CrossEntropyLoss()
    for epoch in tqdm(range(args.epochs), desc="Final Training"):
        train_one_epoch(model_final, all_loader, optimizer, ce_loss_fn, loss_function, alpha, temperature)
    
    print("Evaluating final model on all data...")
    final_loss, final_accuracy = evaluate(model_final, all_loader, ce_loss_fn, loss_function, alpha, temperature)
    print(f"Final model - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")

    torch.save(model_final.state_dict(), args.output)
    print(f"Saved final model to {args.output}")


if __name__ == "__main__":
    args = parse_args()
    run_id = os.environ.get("SLURM_JOB_ID", os.urandom(4).hex())
    mode = "offline" if os.environ.get("WANDB_MODE") == "offline" else "online"
    wandb.init(
        project=WANDB_PROJECT_NAME,
        config=args,
        mode=mode
    )
    wandb.run.name = f"{WANDB_PROJECT_NAME}-{run_id}"
    main(args)
