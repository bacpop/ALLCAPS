import os
import argparse
from tqdm import tqdm

import torch
import numpy as np
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModelForMaskedLM


DEFAULT_MODEL = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_STRIDE = 0.75  # 25% overlap
DEFAULT_MAX_LEN = 20_000  # Max length for CONTIGS, to avoid a sparse matrix upon padding


def chunk_sequence(seq, chunk_size=512, stride=256):
    return [seq[i:i + chunk_size] for i in range(0, len(seq) - chunk_size + 1, stride)]


def embed_chunks(chunks, tokenizer, model, device, max_length):
    inputs = tokenizer(
        chunks,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # (B, T, D)
        pooled = last_hidden.mean(dim=1)         # (B, D)
    return pooled.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True, help="Input FASTA file")
    parser.add_argument("--out_dir", required=True, help="Output directory to store one .npy per sequence")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_name", default=DEFAULT_MODEL, help="Model name or path")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size for embedding")
    parser.add_argument("--stride_ratio", type=float, default=DEFAULT_STRIDE, help="Stride ratio for chunking")
    parser.add_argument("--seq_max_len", type=int, default=DEFAULT_MAX_LEN, help="Maximum sequence length for the CONTIGS")
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name).to(args.device)

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
        seq = str(record.seq)[:args.seq_max_len]

        chunks = chunk_sequence(seq, args.chunk_size, stride)
        if len(chunks) == 0:
            print(f"Skipping {public_name} due to no valid chunks.")
            continue

        pooled = embed_chunks(chunks, tokenizer, model, args.device, max_length)  # shape (L, D)
        np.save(os.path.join(args.out_dir, f"{public_name}.npy"), pooled.numpy())  # tensor [L, D]

    print(f"Saved {total} chunked sequences to {args.out_dir}")

if __name__ == "__main__":
    main()