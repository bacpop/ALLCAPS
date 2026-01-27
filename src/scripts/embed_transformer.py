import os
import argparse
from tqdm import tqdm

import torch
import numpy as np
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModelForMaskedLM

from .consts import (
    DEFAULT_MODEL, DEFAULT_CHUNK_SIZE,
    DEFAULT_STRIDE_RATIO, DEFAULT_MAX_LEN
)
from .utils import chunk_sequence, embed_chunks


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

    # NOTE: Make sure any changes is synced with the query processing script
    model_max_length = getattr(tokenizer, "model_max_length", args.chunk_size)
    chunk_size = min(args.chunk_size, model_max_length)
    stride = int(chunk_size * args.stride_ratio)
    print(f"Max length for {args.model_name} is {model_max_length}")
    print(f"Chunk size: {chunk_size}, Stride: {stride}")

    print(f"Loading sequences from {args.fasta}...")
    total = sum(1 for _ in SeqIO.parse(args.fasta, "fasta"))

    for record in tqdm(SeqIO.parse(args.fasta, "fasta"), total=total):
        seq_id = record.id
        sample_name = seq_id.split("__")[0]  # Public ID + Contig ID
        chunks_path = os.path.join(args.out_dir, f"{sample_name}.npy")
        if os.path.exists(chunks_path):
            print(f"Skipping {sample_name} as it already exists.")

        chunks = chunk_sequence(str(record.seq)[:args.seq_max_len], chunk_size, stride)  # TODO This is terribly wrong. WIll fix for the new data (SPAdes).

        if len(chunks) == 0:
            print(f"Skipping {sample_name} due to no valid chunks.")
            continue
        try:
            pooled = embed_chunks(chunks, tokenizer, model, args.device, model_max_length)  # shape (L, D)
            np.save(chunks_path, pooled.numpy())  # tensor [L, D]
        except Exception as e:
            print(f"Error processing {sample_name}: {e}")

    print(f"Saved {total} chunked sequences to {args.out_dir}")

if __name__ == "__main__":
    main()
