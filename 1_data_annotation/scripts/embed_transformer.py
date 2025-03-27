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
    model_name,
    device="cpu",
    batch_size=4,
    max_length=None
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()

    if max_length is None:
        max_length = tokenizer.model_max_length

    all_embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="Batches"):
        batch_seqs = sequences[i : i + batch_size]
        tokenized = tokenizer.batch_encode_plus(
            batch_seqs,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        ).to(device)

        input_ids = tokenized["input_ids"]
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                output_hidden_states=True
            )

        last_hidden = outputs["hidden_states"][-1]
        attn_expanded = attention_mask.unsqueeze(-1)
        summed = torch.sum(last_hidden * attn_expanded, dim=1)
        counts = torch.sum(attn_expanded, dim=1)
        counts = torch.clamp(counts, min=1e-9)
        pooled = summed / counts
        all_embeddings.append(pooled.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings.numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    print("Reading FASTA file...")
    sequences = read_fasta(args.fasta)

    print("Getting transformer embeddings...")
    embeddings = get_transformer_embeddings(
        sequences,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size
    )
    np.save(args.output, embeddings)
    print(f"Saved transformer embeddings to {args.output}")

if __name__ == "__main__":
    main()
