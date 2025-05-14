import torch.nn as nn


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


class TransformerContrastiveHead(nn.Module):
    def __init__(self, input_dim, output_dim=128, nhead=4, num_layers=2, dim_feedforward=512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, output_dim)  # Optional: compress input to d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True  # ensures (B, S, D) input shape
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):  # x: (batch_size, embed_dim)
        # add sequence dimension to use Transformer: (B, 1, D)
        x = self.input_proj(x).unsqueeze(1)
        x = self.encoder(x)  # (B, 1, D)
        return x.squeeze(1)  # back to (B, D)
