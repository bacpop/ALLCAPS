import gzip
import argparse
from functools import partial
from typing import List

import torch
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from transformers import AutoTokenizer, AutoModelForMaskedLM

from ..utils import read_fasta


DEFAULT_CBL_PATH = "/nfs/research/jlees/shorsfield/RARA_pneumo_cps/AtB_All_S_pneumoniae_CBL.fasta"

def calculate_embedding(sequences, max_length=None):
    # Import the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
    model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")

    # Choose the length to which the input sequences are padded. By default, the model max length is chosen
    # TODO decrease it as it impacts the time taken to obtain the embeddings significantly.
    if max_length is None:
        max_length = tokenizer.model_max_length

    # Using dynamic paddiing (padding=True, truncation=True) improves memory usage
    tokens_ids = tokenizer.batch_encode_plus(sequences, return_tensors="pt", padding="max_length", max_length=max_length)["input_ids"]

    attention_mask = tokens_ids != tokenizer.pad_token_id
    torch_outs = model(
        tokens_ids,
        attention_mask=attention_mask,
        encoder_attention_mask=attention_mask,
        output_hidden_states=True
    )
    embeddings = torch_outs['hidden_states'][-1].detach().numpy()
    print(f"Embeddings shape: {embeddings.shape},", f"Embeddings per token: {embeddings}")

    # Add embed dimension axis and compute mean embeddings per sequence
    attention_mask = torch.unsqueeze(attention_mask, dim=-1)
    mean_sequence_embeddings = torch.sum(attention_mask*embeddings, axis=-2)/torch.sum(attention_mask, axis=1)
    print(f"Mean sequence embeddings: {mean_sequence_embeddings}")


def get_sample_accession(record: SeqRecord) -> str:
    return record.id.split(".")[0]


def main(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cbl", default=DEFAULT_CBL_PATH, help="Path to CBL fasta file")
    main(parser.parse_args())