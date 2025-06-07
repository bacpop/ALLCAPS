import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class ContrastiveHead(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class ContrastiveChunkedDataset(Dataset):
    def __init__(self, embeddings_dir, sampled_ids, serotype_labels, capsule_labels):
        """
        embeddings_path: str, path to directory with npy entries of variable length chunked embeddings
        serotype_labels: pd.DataFrame, DataFrame with serotype labels indexed by sample ID.
        """
        self.embedding_dir = embeddings_dir
        self.sample_ids = sampled_ids
        self.serotypes = serotype_labels
        self.is_capsule = capsule_labels 

        assert pd.Series(self.sample_ids).isin([f.split(".")[0] for f in os.listdir(embeddings_dir) if f.endswith('.npy')]).all(), \
            "At least one sample in serotype_labels does not have a corresponding embedding file."

    def __len__(self) -> int:
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        embedding_path = os.path.join(self.embedding_dir, f"{self.sample_ids[idx]}.npy")
        # if not os.path.exists(embedding_path):
        #     raise FileNotFoundError(f"Embedding file not found at {embedding_path}")
        return {
            'sample_id': self.sample_ids[idx],
            'embedding': torch.tensor(np.load(embedding_path), dtype=torch.float32),
            'serotype': self.serotypes[idx],
            'is_capsule': self.is_capsule[idx]
        }


class TransformerContrastiveHead(nn.Module):
    def __init__(self, input_dim, output_dim=128, max_len=64, nhead=4, num_layers=2):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, input_dim)  # TODO Dynamically expand or clamp

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=4 * input_dim,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(input_dim, 2)

        self.project = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        B, L, D = x.size()
        pos = torch.arange(L, device=x.device).unsqueeze(0)  # (1, L)
        x = x + self.pos_embed(pos)
        x = self.encoder(x)  # Encoded (B, L, D)
        x = x.mean(dim=1)  # Pooled (B, D)
        logits = self.classifier(x)  # Classifier output (B, output_dim)
        return logits, F.normalize(self.project(x), dim=1)
