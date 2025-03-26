#!/usr/bin/env python

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import ContrastiveHead


DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10


def contrastive_loss(z_i, z_j, temperature=0.07):
    """
    Simple NT-Xent style contrastive loss for pairs (z_i, z_j).
    """
    batch_size = z_i.shape[0]
    z_i = nn.functional.normalize(z_i, dim=1)
    z_j = nn.functional.normalize(z_j, dim=1)

    logits = z_i @ z_j.t() / temperature
    labels = torch.arange(batch_size, device=z_i.device)
    loss_i = nn.CrossEntropyLoss()(logits, labels)
    loss_j = nn.CrossEntropyLoss()(logits.t(), labels)
    return 0.5 * (loss_i + loss_j)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    args = parser.parse_args()

    device = args.device
    X = np.load(args.embeddings)  # shape (N, D)
    X_torch = torch.from_numpy(X).float().to(device)

    input_dim = X.shape[1]
    contrastive_head = ContrastiveHead(input_dim, output_dim=128).to(device)
    optimizer = optim.Adam(contrastive_head.parameters(), lr=args.lr)

    n = X_torch.shape[0]
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        X_shuffled = X_torch[perm]

        total_loss = 0.0
        num_batches = n // (2 * args.batch_size)

        for b in range(num_batches):
            start = b * args.batch_size
            mid = (b + 1) * args.batch_size
            left_batch = X_shuffled[start:mid]

            start2 = (b + num_batches) * args.batch_size
            end2 = (b + num_batches + 1) * args.batch_size
            right_batch = X_shuffled[start2:end2]

            z_left = contrastive_head(left_batch)
            z_right = contrastive_head(right_batch)

            loss = contrastive_loss(z_left, z_right)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs}, Loss={total_loss/(num_batches+1e-9):.4f}")

    torch.save(contrastive_head.state_dict(), args.output)
    print(f"Saved contrastive head to {args.output}")

if __name__ == "__main__":
    main()
