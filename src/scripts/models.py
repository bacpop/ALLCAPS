import os
import glob
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
    def __init__(self, embeddings_dir, sample_ids, serotype_labels, capsule_labels):
        """
        embeddings_path: str, path to directory with npy entries of variable length chunked embeddings
        serotype_labels: pd.DataFrame, DataFrame with serotype labels indexed by sample ID.
        """
        self.embedding_dir = embeddings_dir
        self.serotypes = serotype_labels
        self.is_capsule = capsule_labels 
        self.sample_ids = sample_ids

        # TODO Validate sub-folders too
        all_embeddings = glob.glob(os.path.join(embeddings_dir, "**/*.npy"))
        file_names = [os.path.basename(f).split(".")[0] for f in all_embeddings]

        missing_samples = set(self.sample_ids) - set(file_names)
        if missing_samples:
            print("{} sample_ids do not have a corresponding embedding file: {}".format(
                len(missing_samples), missing_samples
            ))

    def __len__(self) -> int:
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        subdir = "cbl" if self.is_capsule[idx] else "non-cbl"
        embedding_path = os.path.join(self.embedding_dir, subdir, f"{self.sample_ids[idx]}.npy")
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
        self.classifier = nn.Linear(output_dim, 2)

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
        z = F.normalize(self.project(x), dim=1)
        logits = self.classifier(z)  # Classifier output (B, output_dim)
        return logits, z


class TransformerLRClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, output_dim=128, max_len=64, nhead=4, num_layers=2):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, input_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=4 * input_dim,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cbl_classifier = nn.Linear(output_dim, 2)
        self.serotype_classifier = nn.Linear(output_dim, num_classes)

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
        z = F.normalize(self.project(x), dim=1)
        logits = self.cbl_classifier(z)  # Classifier output (B, output_dim)
        serotype_logits = self.serotype_classifier(z)
        return logits, serotype_logits, z
