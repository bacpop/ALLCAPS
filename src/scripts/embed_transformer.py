import os
import argparse
from tqdm import tqdm

import torch
import numpy as np
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel

from .consts import (
    DEFAULT_MODEL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_STRIDE_RATIO,
    DEFAULT_MAX_LEN,
)
from .logging_config import get_logger
from .utils import chunk_sequence, embed_chunks

logger = get_logger(__name__)

DEFAULT_RECORDS_PER_BATCH = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True, help="Input FASTA file")
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory to store one .npy per sequence",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model_name", default=DEFAULT_MODEL, help="Model name or path"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk size for embedding",
    )
    parser.add_argument(
        "--stride_ratio",
        type=float,
        default=DEFAULT_STRIDE_RATIO,
        help="Stride ratio for chunking",
    )
    parser.add_argument(
        "--seq_max_len",
        type=int,
        default=DEFAULT_MAX_LEN,
        help="Maximum sequence length for the CONTIGS",
    )
    parser.add_argument(
        "--records_per_batch",
        type=int,
        default=DEFAULT_RECORDS_PER_BATCH,
        help="How many FASTA records to fold into one base-model forward.",
    )
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
    # AutoModel returns the encoder only — drops the unused MaskedLM head.
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True).to(
        args.device
    )
    model.eval()  # Critical: Set to evaluation mode for deterministic embeddings

    # NOTE: Make sure any changes is synced with the query processing script
    model_max_length = getattr(tokenizer, "model_max_length", args.chunk_size)
    chunk_size = min(args.chunk_size, model_max_length)
    stride = int(chunk_size * args.stride_ratio)
    logger.info("Max length for %s is %s", args.model_name, model_max_length)
    logger.info("Chunk size: %d, Stride: %d", chunk_size, stride)

    logger.info("Loading sequences from %s...", args.fasta)
    total = sum(1 for _ in SeqIO.parse(args.fasta, "fasta"))

    # ── Cross-record batching ───────────────────────────────
    # Buffer records, fold all their chunks into one forward, slice back.
    pending: list[dict] = []  # {sample_name, chunks_path, chunks}

    def flush(buf):
        if not buf:
            return
        all_chunks: list[str] = []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for entry in buf:
            n = len(entry["chunks"])
            offsets.append((cursor, cursor + n))
            all_chunks.extend(entry["chunks"])
            cursor += n
        try:
            pooled = embed_chunks(
                all_chunks, tokenizer, model, args.device, model_max_length
            )  # (sum_chunks, D)
        except Exception as exc:
            names = ", ".join(e["sample_name"] for e in buf)
            logger.error("Error embedding batch [%s]: %s", names, exc)
            raise
        for entry, (a, b) in zip(buf, offsets):
            np.save(entry["chunks_path"], pooled[a:b].numpy())

    for record in tqdm(SeqIO.parse(args.fasta, "fasta"), total=total):
        seq_id = record.id
        sample_name = seq_id.split("__")[0]  # Public ID + Contig ID
        chunks_path = os.path.join(args.out_dir, f"{sample_name}.npy")
        if os.path.exists(chunks_path):
            logger.info("Skipping %s as it already exists.", sample_name)
            continue

        chunks = chunk_sequence(
            str(record.seq)[: args.seq_max_len], chunk_size, stride
        )

        if not chunks:
            chunks = [str(record.seq)[: args.seq_max_len]]
            logger.warning(
                "Sequence %s shorter than chunk_size; falling back to single chunk.",
                sample_name,
            )

        pending.append(
            {"sample_name": sample_name, "chunks_path": chunks_path, "chunks": chunks}
        )
        if len(pending) >= args.records_per_batch:
            flush(pending)
            pending = []

    flush(pending)

    logger.info("Saved %d chunked sequences to %s", total, args.out_dir)


if __name__ == "__main__":
    main()
