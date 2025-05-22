import argparse
from tqdm import tqdm

import torch
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModelForMaskedLM


DEFAULT_MODEL = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_STRIDE = 0.75  # 25% overlap


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
    parser.add_argument("--output", required=True, help="Output .pt file for all chunked embeddings")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_name", default=DEFAULT_MODEL, help="Model name or path")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size for embedding")
    parser.add_argument("--stride_ratio", type=float, default=DEFAULT_STRIDE, help="Stride ratio for chunking")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name).to(args.device)

    max_length = tokenizer.model_max_length
    chunk_size = min(args.chunk_size, max_length)
    stride = int(chunk_size * args.stride_ratio)
    print(f"Max length for {args.model_name} is {max_length}")
    print(f"Chunk size: {args.chunk_size}, Stride: {stride}")

    embeddings = {}
    print(f"Loading sequences from {args.fasta}...")
    for record in tqdm(SeqIO.parse(args.fasta, "fasta")):
        seq_id = record.id
        seq = str(record.seq)
        chunks = chunk_sequence(seq, args.chunk_size, stride)

        if len(chunks) == 0:
            continue

        pooled = embed_chunks(chunks, tokenizer, model, args.device, max_length)  # shape (L, D)
        embeddings[seq_id] = pooled  # tensor [L, D]

    torch.save(embeddings, args.output)
    print(f"Saved {len(embeddings)} chunked sequences to {args.output}")

if __name__ == "__main__":
    main()