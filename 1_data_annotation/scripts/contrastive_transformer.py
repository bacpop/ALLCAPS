import os
import json
import wandb
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold

from models import TransformerCapsuleClassifier, ContrastiveChunkedDataset
from utils import supervised_contrastive_loss, hierarchical_contrastive_loss


EPS = 1e-9
OUTPUT_DIM = 128
DEFAULT_LR = 2e-5
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_TEMPERATURE = 0.07
DEFAULT_KFOLDS = 5
DEFAULT_WEIGHT_FINE = 1.0
DEFAULT_WEIGHT_COARSE = 0.6
DEFAULT_NUM_LAYERS = 1
DEFAULT_NHEAD = 3
DEFAULT_DIM_FEEDFORWARD = 512
DEFAULT_CONTRASTIVE_LOSS_RATIO = 0.4

MISSING_LABEL = "Non-typeable"  # TODO: Make this a parameter
LABEL_COLUMN = "Serotype"  # TODO: Do sth about it
WANDB_PROJECT_NAME = "contrastive-inference"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--model-params", type=str, default="{}",
                        help="JSON string of model parameters (output_dim, num_layers, nhead, lr, alpha, temperature, etc.)")
    parser.add_argument("--skip-labels", type=str, default="",
                        help="Comma-separated list of labels to skip in training.")
    parser.add_argument("--hierarchical-loss", action="store_true",
                        help="Use weighted (coarse, fine) labels for training.")
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


def collate_fn(batch):
    return pad_sequence([torch.tensor(item) for item in batch], batch_first=True)


def main(args):
    device = args.device
    random_state = args.model_params.get("random_state", 42)
    k_folds = args.model_params.get("k_folds", DEFAULT_KFOLDS)
    temperature = args.model_params.get("temperature", DEFAULT_TEMPERATURE)
    weight_fine = args.model_params.get("weight_fine", DEFAULT_WEIGHT_FINE)
    weight_coarse = args.model_params.get("weight_coarse", DEFAULT_WEIGHT_COARSE)
    num_layers = args.model_params.get("num_layers", DEFAULT_NUM_LAYERS)
    nhead = args.model_params.get("nhead", DEFAULT_NHEAD)
    alpha = args.model_params.get("alpha", DEFAULT_CONTRASTIVE_LOSS_RATIO)
    output_dim = args.model_params.get("output_dim", OUTPUT_DIM)

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
    
    # ----------------- Load Data ---------------------------
    print("Loading data...")
    data = pd.read_csv("your_data.csv")
    sequences = data['sequence'].tolist()
    capsule_labels = data['capsule_label'].values  # TODO binary labels
    serotype_labels = data['serotype_label'].values

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(sequences, capsule_labels)):
        print(f"Fold {fold+1}")

        train_ds = ContrastiveChunkedDataset(np.array(sequences)[train_idx], capsule_labels[train_idx], serotype_labels[train_idx])
        test_ds = ContrastiveChunkedDataset(np.array(sequences)[test_idx], capsule_labels[test_idx], serotype_labels[test_idx])

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)

        model = TransformerCapsuleClassifier(args.model_name).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        ce_loss_fn = nn.CrossEntropyLoss()

        model.train()

        for epoch in range(args.epochs):
            for batch in train_loader:
                encoded = tokenizer(batch['sequence'], padding=True, truncation=True, return_tensors='pt').to('cuda')
                capsule_label = batch['is_capsule'].cuda()
                serotype_label = batch['serotype'].cuda()
                capsule_mask = (capsule_label == 1)

                logits, embeddings = model(encoded['input_ids'], encoded['attention_mask'])

                ce_loss = ce_loss_fn(logits, capsule_label)

                if capsule_mask.sum() > 1:
                    contr_loss = nt_xent_loss(embeddings[capsule_mask], serotype_label[capsule_mask])
                else:
                    contr_loss = torch.tensor(0.0, device=ce_loss.device)

                loss = ce_loss + alpha * contr_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # TODO retrain on all data
    model_final = None
    torch.save(model_final.state_dict(), args.output)
    print(f"Saved final model to {args.output}")


def main(args):
    with open(args.pairs_file, 'r') as f:
        pairs = [line.strip().split(',') for line in f if line.strip()]

    dataset = ContrastiveChunkedDataset(pairs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Infer input dim from first file
    sample = torch.tensor(np.load(pairs[0][0]))
    input_dim = sample.shape[1]

    model = TransformerContrastiveHead(input_dim, args.output_dim).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for xb_left, xb_right in loader:
            xb_left = xb_left.to(args.device)
            xb_right = xb_right.to(args.device)

            z_i = model(xb_left)
            z_j = model(xb_right)

            loss = contrastive_loss(z_i, z_j, temperature=args.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})

    torch.save(model.state_dict(), args.output)
    wandb.save(args.output)


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
