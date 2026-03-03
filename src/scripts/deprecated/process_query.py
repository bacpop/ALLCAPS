import argparse
import os
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from typing import List, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from .models import ModelRegistry, TransformerLRClassifier
from .consts import (
    DEFAULT_MODEL, DEFAULT_CHUNK_SIZE, DEFAULT_MAX_LEN,
    DEFAULT_STRIDE_RATIO, DEFAULT_ENERGY_TEMPERATURE
)
from .utils import chunk_sequence, embed_chunks


EPS = 1e-6
THRESH_CPS = 0.5
DEFAULT_ENERGY_PERCENTILE = 99.0
DEFAULT_ROLLING_STEP = 2000

### Temporary hard-coded stats to skip JSON loading during testing
percentiles_serotype = {
    "93.0": -8.152,
    "95.0": -8.368741035461426,
    "99.0": -6.334590911865234,
    "99.5": -5.951267242431641
}
###

def energy_score(logits, temperature=1.0) -> float:
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    energy = -temperature * torch.logsumexp(logits / temperature, dim=-1)
    return float(energy.item())


def transformer_embedding(  # TODO batch this 
    tokenizer: AutoTokenizer,
    base_model: AutoModelForMaskedLM,
    logistic_model: TransformerLRClassifier,
    sequence: str,
    device: str = "cuda",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    stride_ratio: float = DEFAULT_STRIDE_RATIO,
    step: int = 1000,
    max_length: int = DEFAULT_MAX_LEN,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Given a list of sequences, chunk and embed them using a pretrained transformer,
    then feed through the trained classifier to get final embeddings.
    
    Inputs:
        tokenizer: Pretrained tokenizer for the base model
        base_model: Pretrained base transformer model
        logistic_model: Trained TransformerLRClassifier model
        sequences: List of input sequences (strings)
        device: Device to run computations on ("cuda" or "cpu")
        chunk_size: Size of each chunk for embedding
        stride_ratio: Ratio of chunk size to use as stride
        step: Step size for sliding window. Do not mistake for stride.
        max_length: Maximum length of sequences to process.
                    Longer sequences are chopped into `max_length` proceeding with `step`.
    
    Returns: np.ndarray of shape (len(sequences), output_dim)
    
    """
    # Use tokenizer.model_max_length if available, otherwise fallback to provided max_length
    model_max_length = getattr(tokenizer, "model_max_length", chunk_size)
    chunk_size = min(chunk_size, model_max_length)
    stride = int(chunk_size * stride_ratio)

    all_cbl_logits, all_serotype_logits, all_embeddings = [], [], []
    start_indices = [i for i in range(0, len(sequence), step) if i + max_length <= len(sequence)]
    if not start_indices:
        start_indices = [0]
    
    # Iterate over the sequence, embedding `max_length` sequences at a time, to get the most confidente prediction
    for i in start_indices:
        candidate = sequence[i:i + max_length]

        # Chunk the sequence
        chunks = chunk_sequence(candidate, chunk_size, stride)
        if not chunks:
            chunks = [candidate]
        
        # Get raw chunked embeddings from pre-trained transformer
        pooled = embed_chunks(chunks, tokenizer, base_model, device, model_max_length)  # shape (L, D)

        # Feed through the full logistic transformer model (like training pipeline)
        with torch.no_grad():
            inputs = pooled.unsqueeze(0).to(device)  # (1, L, D)
            cbl_logits, serotype_logits, embedding = logistic_model(inputs)  # Full model forward pass
            embedding = embedding.squeeze(0).cpu().numpy()  # (output_dim,) - final projected embedding

        all_cbl_logits.append(cbl_logits.cpu().numpy())
        all_serotype_logits.append(serotype_logits.cpu().numpy())
        all_embeddings.append(embedding)

    return all_cbl_logits, all_serotype_logits, all_embeddings


def main(args):
    device = args.device
    chunk_size = args.model_params.get("chunk_size", DEFAULT_CHUNK_SIZE)
    stride_ratio = args.model_params.get("stride_ratio", DEFAULT_STRIDE_RATIO)

    max_length = args.model_params.get("max_length", DEFAULT_MAX_LEN)
    rolling_step = args.model_params.get("rolling_step", DEFAULT_ROLLING_STEP)

    cbl_threshold = THRESH_CPS

    # Prepare energy temperature and threshold (tau)
    resolved_temperature = float(args.energy_temperature)
    tau_serotype: Optional[float] = percentiles_serotype.get(str(args.energy_percentile), None)
    assert tau_serotype is not None, "tau_serotype must be set"
    # print(f"Loaded tau_serotype={tau_serotype:.6f} at T={resolved_temperature} from {args.energy_thresholds_json}")

    # Set seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Loading the {args.base_model} base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    base_model = AutoModelForMaskedLM.from_pretrained(args.base_model, trust_remote_code=True).to(device)
    base_model.eval()  # Set to eval mode for deterministic behavior
    
    print("Loading the transformer and logistic regression model...")
    model_save_dict = torch.load(args.model_path, map_location=device)
    model_config = model_save_dict['model_config']
    idx_to_serotype = {v: k for k, v in model_save_dict['serotype_to_idx'].items()}
    
    # Initialize model with saved configuration
    logistic_model = ModelRegistry.get_model_class(args.head_model) \
        .from_config(model_config) \
        .to(device)
    logistic_model.load_state_dict(model_save_dict['model_state_dict'])
    logistic_model.eval()
    
    print("Processing queries...")
    results = dict()
    query_sequences = list(SeqIO.parse(args.query, "fasta"))
    for record in tqdm(query_sequences, desc="Processing queries"):
        print(f"Processing query: {record.id}...")
        query_cbl_logits, query_serotype_logits, query_embedding = transformer_embedding(
            tokenizer=tokenizer,
            base_model=base_model,
            logistic_model=logistic_model,
            sequence=str(record.seq),
            device=device,
            chunk_size=chunk_size,
            stride_ratio=stride_ratio,
            step=rolling_step,
            max_length=max_length,
        )

        # Select among candidate windows: keep those with is_capsule=True and pick highest serotype confidence
        best_idx = None
        best_conf = -np.inf
        # Compute per-candidate serotype confidence; filter to capsulated
        for idx in range(len(query_serotype_logits)):
            cbl_vec = np.asarray(query_cbl_logits[idx]).squeeze()
            sero_vec = np.asarray(query_serotype_logits[idx]).squeeze()
            # Capsule decision for this candidate (consistent with existing logic)
            cbl_probs = torch.sigmoid(torch.tensor(cbl_vec)).numpy()
            is_capsulated = bool(cbl_probs[1] > cbl_threshold)
            if not is_capsulated:
                continue
            # Serotype confidence = max softmax probability
            sero_conf = float(torch.softmax(torch.tensor(sero_vec), dim=-1).max().item())
            if sero_conf > best_conf:
                best_conf = sero_conf
                best_idx = idx

        # Fallback: if no capsulated candidates, use highest serotype confidence overall
        if best_idx is None:
            for idx in range(len(query_serotype_logits)):
                sero_vec = np.asarray(query_serotype_logits[idx]).squeeze()
                sero_conf = float(torch.softmax(torch.tensor(sero_vec), dim=-1).max().item())
                if sero_conf > best_conf:
                    best_conf = sero_conf
                    best_idx = idx

        # Extract results for the selected candidate
        assert best_idx is not None, f"No candidate window selected in {record.id}"
        sel_cbl_logits = np.asarray(query_cbl_logits[best_idx]).squeeze()
        sel_serotype_logits = np.asarray(query_serotype_logits[best_idx]).squeeze()
        sel_embedding = np.asarray(query_embedding[best_idx])
        cbl_predictions = torch.sigmoid(torch.tensor(sel_cbl_logits)).numpy()
        is_cbl = bool(cbl_predictions[1] > cbl_threshold)

        # Serotype energy at configured temperature (novelty score)
        e_sero = energy_score(sel_serotype_logits, temperature=resolved_temperature)
        is_novel = bool(e_sero > tau_serotype)  # TODO should this also depend on is_cbl?

        results[record.id] = {
            "serotype_logits": sel_serotype_logits,
            "embedding": sel_embedding,
            "is_cbl": is_cbl,
            "is_novel_serogroup": is_novel,
            "serotype_confidence": round(best_conf, 3),
            "novelty_confidence": round(e_sero, 3),
        }
    
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df["pred_argmax"] = results_df["serotype_logits"].apply(
        lambda x: idx_to_serotype[np.argmax(torch.softmax(torch.tensor(x), dim=-1).numpy())]
    )
    # Save query embeddings
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.output_dir, "query_embeddings.npz"),
        record_ids=results_df.index.to_numpy(),
        embeddings=np.stack(results_df["embedding"].values)
    )
    # Save full results
    results_df \
        .drop(columns=["embedding", "serotype_logits"]) \
        .to_csv(os.path.join(args.output_dir, "query_results.csv"))


def parse_args():
    parser = argparse.ArgumentParser(description="Novel detection script.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the output files.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to use for computation.")
    parser.add_argument("--query", type=str, required=True,
                        help="Path to the query FASTA file.")
    parser.add_argument("--model_params", type=str, default="{}",
                        help="JSON string of model parameters")
    parser.add_argument("--base_model", type=str, default=DEFAULT_MODEL,
                        help="Name of the base transformer model to use for initial inference.")
    parser.add_argument("--head_model", type=str, default="transformer_lr_classifier",
                        help="Name of the head model to use.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model.")
    parser.add_argument("--energy_temperature", type=float, default=DEFAULT_ENERGY_TEMPERATURE,
                        help="Temperature T for energy computation")
    parser.add_argument("--energy_percentile", type=float, default=DEFAULT_ENERGY_PERCENTILE,
                        help="Percentile (e.g., 99) over ID energies to set tau_serotype")
    # parser.add_argument("--energy_thresholds_json", type=str, required=True,
    #                     help="JSON file containing precomputed tau_serotype and temperature")
    parser.add_argument("--query_mode", default="default", choices=["default", "fast"],
                        help="Query processing mode. Default is 'default' which uses full model inference. "
                             "Fast mode skips the alignment-based locus cutter. "
                             "Faster runtime is possible by setting smaller values for stride_ratio.")
    args = parser.parse_args()
    
    try:
        args.model_params = json.loads(args.model_params)
        if not isinstance(args.model_params, dict):
            print("Model parameters should be a JSON object.")
            args.model_params = {}
    except json.JSONDecodeError:
        print("Error parsing model parameters JSON string.")
        args.model_params = {}
    finally:
        print("Model parameters:", args.model_params)

    return args


if __name__ == "__main__":
    main(parse_args())
