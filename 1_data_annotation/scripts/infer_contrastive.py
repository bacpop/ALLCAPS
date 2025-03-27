#!/usr/bin/env python

import argparse
import numpy as np
import torch

from models import ContrastiveHead

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("Loading embeddings...")
    device = args.device
    X = np.load(args.embeddings)  # shape (N, D)
    X_torch = torch.from_numpy(X).float().to(device)

    input_dim = X.shape[1]
    contrastive_head = ContrastiveHead(input_dim, output_dim=128).to(device)
    contrastive_head.load_state_dict(torch.load(args.head, map_location=device))
    contrastive_head.eval()

    with torch.no_grad():
        z = contrastive_head(X_torch)
    z_np = z.cpu().numpy()

    np.save(args.output, z_np)
    print(f"Saved contrastive embeddings to {args.output}")

if __name__ == "__main__":
    main()
