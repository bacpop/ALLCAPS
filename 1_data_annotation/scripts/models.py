import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoModel


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
    def __init__(self, embeddings_path, capsule_labels, serotype_labels):
        """
        embeddings_path: str, path to .pt or .npz with {'sample_id': [L, D]} entries
        capsule_labels: list of capsule labels (binary)
        serotype_labels: list of serotype labels
        """
        if embeddings_path.endswith(".pt"):
            self.embeddings = torch.load(embeddings_path)
        elif embeddings_path.endswith(".npz"):
            self.embeddings = {k: torch.tensor(v) for k, v in np.load(embeddings_path, allow_pickle=True).items()}
        else:
            raise ValueError("Unsupported format. Use .pt or .npz")

        self.capsule_labels = capsule_labels
        self.serotype_labels = serotype_labels
        assert len(self.embeddings) == len(capsule_labels) == len(serotype_labels), \
            "Mismatch in number of samples between embeddings and labels."

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'is_capsule': self.capsule_labels[idx],
            'serotype': self.serotype_labels[idx]
        }


class TransformerCapsuleClassifier(nn.Module):
    def __init__(self, transformer_name, extra_layers=2, num_heads=8, hidden_dim=128):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(transformer_name)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer.config.hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=extra_layers)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        embeddings = self.transformer(input_ids, attention_mask).last_hidden_state
        encoded_emb = self.encoder(embeddings.permute(1,0,2)).permute(1,0,2)
        pooled_emb = encoded_emb.mean(dim=1)
        logits = self.classifier(pooled_emb)
        return logits, pooled_emb


class TransformerContrastiveHead(nn.Module):
    def __init__(self, input_dim, output_dim=128, max_len=128, nhead=4, num_layers=2):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, input_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=4 * input_dim,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.project = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        B, L, D = x.size()
        pos = torch.arange(L, device=x.device).unsqueeze(0)  # (1, L)
        x = x + self.pos_embed(pos)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return F.normalize(self.project(x), dim=1)