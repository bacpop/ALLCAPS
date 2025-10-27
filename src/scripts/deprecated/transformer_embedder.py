import os
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from models import TransformerContrastiveHead

from consts import (
    DEFAULT_MODEL, DEFAULT_CHUNK_SIZE, DEFAULT_STRIDE_RATIO, DEFAULT_MAX_LEN,
    DEFAULT_EMBEDDING_DIM, DEFAULT_OUTPUT_DIM, DEFAULT_NHEAD, DEFAULT_NUM_LAYERS
)

class TransformerEmbedder:
    def __init__(self, head_model_path, nt_model_name=DEFAULT_MODEL, device="cuda", 
                 embedding_dim=DEFAULT_EMBEDDING_DIM, output_dim=DEFAULT_OUTPUT_DIM, 
                 nhead=DEFAULT_NHEAD, num_layers=DEFAULT_NUM_LAYERS, 
                 chunk_size=DEFAULT_CHUNK_SIZE, stride_ratio=DEFAULT_STRIDE_RATIO):
        
        self.device = torch.device(device)
        self.head_model_path = head_model_path
        self.nt_model_name = nt_model_name
        self.chunk_size = chunk_size
        self.stride_ratio = stride_ratio

        self.tokenizer = None  # To be initialized in init_transformer
        self.nt_model = None
        self._load_head_model(embedding_dim, output_dim, nhead, num_layers)

    def init_transformer(self):
        """
        Initialize the Nucleotide Transformer model and tokenizer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.nt_model_name)
        self.nt_model = AutoModelForMaskedLM.from_pretrained(self.nt_model_name).to(self.device)
        
        self.max_length = self.tokenizer.model_max_length
        self.chunk_size = min(self.chunk_size, self.max_length)
        self.stride = int(self.chunk_size * self.stride_ratio)

    def _load_head_model(self, embedding_dim, output_dim, nhead, num_layers):
        if self.head_model:
            print("Head model already loaded, skipping.")
            return
        if os.path.exists(self.head_model_path):
            self.head_model = TransformerContrastiveHead(
                input_dim=embedding_dim,
                output_dim=output_dim,
                nhead=nhead,
                num_layers=num_layers
            ).to(self.device)
            self.head_model.load_state_dict(torch.load(self.head_model_path, map_location=self.device))
            self.head_model.eval()
            print(f"Loaded head model from {self.head_model_path}")
        else:
            print(f"Head model path {self.head_model_path} not found.")

    def _chunk_sequence(self, seq):
        return [seq[i:i + self.chunk_size] for i in range(0, len(seq) - self.chunk_size + 1, self.stride)]

    def embed_sequence_chunks(self, seq) -> torch.Tensor:
        """
        Embed a sequence by chunking it into smaller pieces.
        Returns a tensor of shape (B_i, D) where B_i is the number of chunks and D is the embedding dimension.
        """
        if not self.tokenizer or not self.nt_model:
            raise ValueError("Transformer model and tokenizer must be initialized.")
        
        chunks = self._chunk_sequence(seq)
        if not chunks:
            raise ValueError("Input sequence is too short to chunk.")
        inputs = self.tokenizer(
            chunks,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.nt_model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]  # (B_i, T, D)
            pooled = last_hidden.mean(dim=1)         # (B_i, D)
        return pooled.cpu().numpy()

    def _infer_contrastive_embeddings(self, chunks):
        """
        Feed the pooled chunks through the contrastive head model to get final embeddings.
        Returns a tensor of shape (1, output_dim).
        """
        with torch.no_grad():
            logits, embedding = self.head_model(chunks.unsqueeze(0))
            embedding = embedding.squeeze(0).cpu().numpy()