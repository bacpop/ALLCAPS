import os
import argparse
from tqdm import tqdm

import torch
import numpy as np
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModelForMaskedLM

from consts import (
    DEFAULT_MODEL, DEFAULT_CHUNK_SIZE,
    DEFAULT_STRIDE_RATIO, DEFAULT_MAX_LEN
)
from utils import chunk_sequence, embed_chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True, help="Input FASTA file")
    parser.add_argument("--out_dir", required=True, help="Output directory to store one .npy per sequence")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_name", default=DEFAULT_MODEL, help="Model name or path")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size for embedding")
    parser.add_argument("--stride_ratio", type=float, default=DEFAULT_STRIDE_RATIO, help="Stride ratio for chunking")
    parser.add_argument("--seq_max_len", type=int, default=DEFAULT_MAX_LEN, help="Maximum sequence length for the CONTIGS")
    args = parser.parse_args()

    # Set seeds for reproducible embeddings
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name, trust_remote_code=True).to(args.device)
    model.eval()  # Critical: Set to evaluation mode for deterministic embeddings

    max_length = tokenizer.model_max_length
    chunk_size = min(args.chunk_size, max_length)
    stride = int(chunk_size * args.stride_ratio)
    print(f"Max length for {args.model_name} is {max_length}")
    print(f"Chunk size: {args.chunk_size}, Stride: {stride}")

    print(f"Loading sequences from {args.fasta}...")
    total = sum(1 for _ in SeqIO.parse(args.fasta, "fasta"))
    for record in tqdm(SeqIO.parse(args.fasta, "fasta"), total=total):
        seq_id = record.id
        public_name = seq_id.split("__")[0]
        seq = str(record.seq)[:args.seq_max_len]  # TODO Is it ok to truncate here?
        # TODO filter somewhere else in a resuable module

        chunks = chunk_sequence(seq, args.chunk_size, stride)
        if len(chunks) == 0:
            print(f"Skipping {public_name} due to no valid chunks.")
            continue

        pooled = embed_chunks(chunks, tokenizer, model, args.device, max_length)  # shape (L, D)
        np.save(os.path.join(args.out_dir, f"{public_name}.npy"), pooled.numpy())  # tensor [L, D]

    print(f"Saved {total} chunked sequences to {args.out_dir}")

if __name__ == "__main__":
    main()
