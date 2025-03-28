#!/usr/bin/env python

import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from Bio import SeqIO
from tqdm import tqdm


DEFAULT_MODEL = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"


def read_fasta(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
    return sequences


def get_transformer_embeddings(
    sequences,
    device,
    model_name=DEFAULT_MODEL,
    max_length=None,
    chunk_size=1000
):
    """
    Extract embeddings for each sequence using a 'sliding window' approach:
      1. Split a long sequence into overlapping chunks of size `chunk_size`,
         overlapping by `overlap` tokens.
      2. Pass each chunk through the model separately.
      3. Average (mean-pool) the chunk-level embeddings to produce one final
         embedding per original sequence.

    Returns:
        np.ndarray of shape (num_sequences, hidden_dim):
          The final average-pooled embeddings for each original sequence.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()

    if max_length is None:
        max_length = tokenizer.model_max_length

    chunk_size = min(chunk_size, max_length)
    overlap = chunk_size // 4  # 25% overlap  TODO try 50

    all_sequence_embeddings = []

    for seq in tqdm(sequences):
        tokens = tokenizer.encode(seq, add_special_tokens=True)
        # tokens is now a list of token IDs. Possibly very long.

        if len(tokens) <= chunk_size:  # short enough, no need to chunk, still need to clamp
            tokens_tensor = torch.tensor([tokens[:max_length]], device=device)
            
            with torch.no_grad():
                attention_mask = (tokens_tensor != tokenizer.pad_token_id).long()
                outputs = model(
                    tokens_tensor, 
                    attention_mask=attention_mask, 
                    output_hidden_states=True
                )
            last_hidden = outputs["hidden_states"][-1]  # (1, seq_len, hidden_dim)

            # Mean-pool (ignoring any padded positions)
            seq_len = tokens_tensor.shape[1]
            valid_mask = attention_mask.unsqueeze(-1)  # (1, seq_len, 1)
            sum_vec = (last_hidden * valid_mask).sum(dim=1)
            count_vec = valid_mask.sum(dim=1).clamp(min=1e-9)
            pooled = (sum_vec / count_vec).squeeze(0)  # shape (hidden_dim,)

            all_sequence_embeddings.append(pooled.cpu())
        else:
            chunk_embeddings = []
            start = 0
            stride = chunk_size - overlap  # the window to slide by 
            while start < len(tokens):
                end = start + chunk_size
                chunk_ids = tokens[start:end]

                tokens_tensor = torch.tensor([chunk_ids], device=device)
                with torch.no_grad():
                    attention_mask = (tokens_tensor != tokenizer.pad_token_id).long()
                    outputs = model(
                        tokens_tensor,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                last_hidden = outputs["hidden_states"][-1]  # shape (1, chunk_len, hidden_dim)

                # Mean-pool over chunk_len
                c_len = tokens_tensor.shape[1]
                valid_mask = attention_mask.unsqueeze(-1)  # shape (1, chunk_len, 1)
                sum_vec = (last_hidden * valid_mask).sum(dim=1)
                count_vec = valid_mask.sum(dim=1).clamp(min=1e-9)
                pooled_chunk = sum_vec / count_vec  # shape (1, hidden_dim)

                chunk_embeddings.append(pooled_chunk.squeeze(0).cpu())

                start += stride
                if start >= len(tokens):
                    break

            # Now average all chunk embeddings for this sequence
            chunk_embeddings = torch.stack(chunk_embeddings, dim=0)  # (num_chunks, hidden_dim)
            seq_embedding = chunk_embeddings.mean(dim=0)  # (hidden_dim,)
            all_sequence_embeddings.append(seq_embedding)
    
    return torch.stack(all_sequence_embeddings, dim=0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("Reading FASTA file...")
    sequences = read_fasta(args.fasta)

    print("Getting transformer embeddings...")
    embeddings = get_transformer_embeddings(
        sequences,
        device=args.device,
        model_name=args.model_name,
    )
    np.save(args.output, embeddings)
    print(f"Saved transformer embeddings to {args.output}")

if __name__ == "__main__":
    main()
