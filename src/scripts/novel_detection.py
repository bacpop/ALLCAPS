import argparse
import os
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from models import ModelRegistry, TransformerLRClassifier
from consts import (
    DEFAULT_MIN_SEROGROUP_SIZE, DEFAULT_MODEL, DEFAULT_CHUNK_SIZE,
    DEFAULT_STRIDE_RATIO, DEFAULT_MAX_LEN, DEFAULT_SEP,
    DEFAULT_BATCH_SIZE, CONTIG_SEP
)
from utils import chunk_sequence, embed_chunks, load_data


EPS = 1e-6
THRESH_CPS = 0.5  # TODO clean up and document and verify and what the fuck
THRESH_NONCPS = 0.1
NORM_NONCBL_PPF = 0.95
THRESH_BETA = 0.98


def energy_score(logits, temperature=1.0) -> float:
    energy = -temperature * torch.logsumexp(logits / temperature, dim=-1)  # TODO verify
    return energy.item()


def transformer_embedding(  # TODO batch this 
    tokenizer: AutoTokenizer,
    base_model: AutoModelForMaskedLM,
    logistic_model: TransformerLRClassifier,
    sequences: List[str],
    device: str = "cuda",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    stride_ratio: float = DEFAULT_STRIDE_RATIO,
    max_length: int = DEFAULT_MAX_LEN,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a list of sequences, chunk and embed them using a Nucleotide Transformer,
    then feed through the trained TransformerLRClassifier to get final embeddings.
    Returns: np.ndarray of shape (len(sequences), output_dim)
    
    This function replicates the training/inference pipeline exactly.
    """
    # Use tokenizer.model_max_length if available, otherwise fallback to provided max_length
    tok_max_len = getattr(tokenizer, "model_max_length", max_length)
    chunk_size = min(chunk_size, tok_max_len)
    stride = int(chunk_size * stride_ratio)

    all_cbl_logits, all_serotype_logits = [], []
    all_embeddings = []
    for seq in sequences:
        # Chunk the sequence
        chunks = chunk_sequence(seq[:45000], chunk_size, stride)
        if not chunks:
            # print(f"Skipping sequence due to no valid chunks: {seq[:30]}...")
            chunks = [seq]
        
        # Get raw chunked embeddings from NT model
        pooled = embed_chunks(chunks, tokenizer, base_model, device, tok_max_len)  # shape (L, D)    

        # Feed through the full logistic transformer model (like training pipeline)
        with torch.no_grad():
            inputs = pooled.unsqueeze(0).to(device)  # (1, L, D)
            cbl_logits, serotype_logits, embedding = logistic_model(inputs)  # Full model forward pass
            embedding = embedding.squeeze(0).cpu().numpy()  # (output_dim,) - final projected embedding

        all_cbl_logits.append(cbl_logits.cpu().numpy())
        all_serotype_logits.append(serotype_logits.cpu().numpy())
        all_embeddings.append(embedding)

    return np.stack(all_cbl_logits), np.stack(all_serotype_logits), np.stack(all_embeddings)


def head_model_inference(
    model: TransformerLRClassifier,
    X: np.ndarray,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on the logistic regression head model.
    X: np.ndarray of shape (N, D) where N is number of samples and D is embedding dimension.
    Returns: Tuple of (cbl_logits, serotype_logits)
    
    Note: X should be the final projected embeddings (z) from the model, not raw embeddings.
    """
    device = next(model.parameters()).device
    model.eval()
    
    all_cbl_logits = []
    all_serotype_logits = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(X), batch_size)):
            batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            # Apply the classifier layers directly to the embeddings (like in serotype_classifier.py)
            cbl_logits = model.cbl_classifier(batch)
            serotype_logits = model.serotype_classifier(batch)
            
            all_cbl_logits.append(cbl_logits.cpu().numpy())
            all_serotype_logits.append(serotype_logits.cpu().numpy())
    
    return np.concatenate(all_cbl_logits), np.concatenate(all_serotype_logits)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    chunk_size = args.model_params.get("chunk_size", DEFAULT_CHUNK_SIZE)
    stride_ratio = args.model_params.get("stride_ratio", DEFAULT_STRIDE_RATIO)
    max_length = args.model_params.get("max_length", DEFAULT_MAX_LEN)
    batch_size = args.model_params.get("batch_size", DEFAULT_BATCH_SIZE)
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    ###

    thresholds = list(map(float, map(str.strip, args.thresholds.split(","))))
    assert len(thresholds) == 4, "Four thresholds are required."
    assert all(0 < t < 1 for t in thresholds), "Thresholds must be between 0 and 1."

    print(f"Loading the {args.base_model} base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    base_model = AutoModelForMaskedLM.from_pretrained(args.base_model, trust_remote_code=True).to(device)
    base_model.eval()  # Set to eval mode for deterministic behavior
    
    print("Loading the transformer and logistic regression model...")
    model_path = os.path.join(args.output_dir, "transformer_model.pth")
    model_save_dict = torch.load(model_path, map_location=device)
    model_config = model_save_dict['model_config']
    idx_to_serotype = {v: k for k, v in model_save_dict['serotype_to_idx'].items()}
    
    # Initialize model with saved configuration
    logistic_model = ModelRegistry.get_model_class(args.head_model) \
        .from_config(model_config) \
        .to(device)
    logistic_model.load_state_dict(model_save_dict['model_state_dict'])
    logistic_model.eval()

    # Prepare energy temperature and threshold (tau) for query mode
    resolved_temperature = args.energy_temperature
    from typing import Optional
    tau_serotype: Optional[float] = None
    if not args.embeddings:
        # Query mode: determine tau from JSON or CSV
        if args.energy_thresholds_json:
            with open(args.energy_thresholds_json, 'r') as f:
                thr = json.load(f)
            # Prefer explicit tau_serotype, else pick percentile value
            if 'temperature' in thr:
                resolved_temperature = float(thr['temperature'])
            if 'tau_serotype' in thr:
                tau_serotype = float(thr['tau_serotype'])
            elif 'percentiles_serotype' in thr:
                perc_map = thr['percentiles_serotype']
                key_candidates = [str(args.energy_percentile), str(int(args.energy_percentile))]
                found = False
                for k in key_candidates:
                    if k in perc_map:
                        tau_serotype = float(perc_map[k])
                        found = True
                        break
                if not found:
                    raise ValueError("Thresholds JSON missing matching percentile for energy.")
            else:
                raise ValueError("Thresholds JSON missing tau_serotype or percentiles_serotype.")
            print(f"Loaded tau_serotype={tau_serotype:.6f} at T={resolved_temperature} from {args.energy_thresholds_json}")
        elif args.id_energies_csv:
            df_e = pd.read_csv(args.id_energies_csv)
            # Ensure numeric energies
            if 'energy_serotype' not in df_e.columns:
                raise ValueError("ID energies CSV must contain 'energy_serotype' column.")
            energies = pd.to_numeric(df_e['energy_serotype'], errors='coerce').to_numpy(dtype=float)
            energies = energies[~np.isnan(energies)]
            if 'Is_capsule' in df_e.columns:
                caps = df_e['Is_capsule'].astype(bool).to_numpy()
                # Align mask length if any rows were NaN; fallback to not masking when lengths mismatch
                if len(caps) == len(df_e):
                    # Apply mask before NaN filtering: recompute mask indices
                    valid_mask = ~pd.to_numeric(df_e['energy_serotype'], errors='coerce').isna().to_numpy()
                    if valid_mask.sum() == len(energies):
                        caps = caps[valid_mask]
                        energies = energies[caps]
            if energies.size == 0:
                raise ValueError("No valid energy values found in ID energies CSV after filtering.")
            tau_serotype = float(np.percentile(energies.astype(float), float(args.energy_percentile)))
            print(f"Computed tau_serotype={tau_serotype:.6f} from ID energies (p={args.energy_percentile})")
        else:
            # Should not happen due to arg checks, but guard anyway
            raise ValueError("In query mode, provide --id_energies_csv or --energy_thresholds_json.")

    results = dict()
    if args.embeddings:
        print("Loading embeddings and labels...")
        embeddings, labels = load_data(args.embeddings, args.labels, sep=DEFAULT_SEP)
        print(f"Loaded embeddings shape: {embeddings.shape}")
        print(f"Expected embedding dimension: {model_config['output_dim']}")

        sample_size = 10000
        print(f"Sampling {sample_size} embeddings from {len(embeddings)} total.")
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings, labels = embeddings[indices], labels.iloc[indices]
        labels['sample_id'] = labels.index + CONTIG_SEP + labels['Contig_ID']
        print("Running inference on embeddings...") 
        # Note: embeddings are expected to be final projected embeddings (z) from the model
        cbl_logits, serotype_logits = head_model_inference(logistic_model, embeddings, batch_size=batch_size)
        for sample in tqdm(range(len(embeddings))):
            sample_id = labels.iloc[sample]['sample_id']
            results[sample_id] = {
                "ground_truth": labels.iloc[sample]['Serotype'],
                "cbl_logits": cbl_logits[sample],
                "serotype_logits": serotype_logits[sample],
                "embedding": embeddings[sample],
                "is_cbl": torch.sigmoid(torch.tensor(cbl_logits[sample]))[1].item() > thresholds[0],  # CBL probability
            }

    else:
        print("Processing queries...")
        query_sequences = list(SeqIO.parse(args.query, "fasta"))
        # Ensure tau is resolved for type checkers and runtime
        assert tau_serotype is not None, "tau_serotype must be set in query mode"
        for record in tqdm(query_sequences, desc="Processing queries"):
            print(f"Processing query: {record.id}...")
            query_cbl_logits, query_serotype_logits, query_embedding = transformer_embedding(
                tokenizer=tokenizer,
                base_model=base_model,
                logistic_model=logistic_model,
                sequences=[str(record.seq)],
                device=device,
                chunk_size=chunk_size,
                stride_ratio=stride_ratio,
                max_length=max_length,
            )
            
            # Extract results for this single sequence
            query_cbl_logits = query_cbl_logits[0, 0]  # (1, 1, 2) -> (2,)
            query_serotype_logits = query_serotype_logits[0, 0]  # (1, 1, num_classes) -> (num_classes,)
            query_embedding = query_embedding[0]  # (1, output_dim) -> (output_dim,)
            cbl_predictions = torch.sigmoid(torch.tensor(query_cbl_logits)).numpy()
            is_cbl = bool(cbl_predictions[1] > thresholds[0])
            # Serotype energy at configured temperature
            e_sero = energy_score(query_serotype_logits, temperature=resolved_temperature)
            is_novel = bool(is_cbl and (float(e_sero) > float(tau_serotype)))
            results[record.id] = {
                "cbl_logits": query_cbl_logits,
                "serotype_logits": query_serotype_logits,
                "embedding": query_embedding,
                "is_cbl": is_cbl,
                "energy_serotype": float(e_sero),
                "is_novel_serogroup": is_novel,
            }
    
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df["pred_argmax"] = results_df["serotype_logits"].apply(
        lambda x: idx_to_serotype[np.argmax(torch.softmax(torch.tensor(x), dim=-1).numpy())]
    )
    # If query mode, add tau and temperature for traceability
    if not args.embeddings:
        results_df["tau_serotype"] = tau_serotype
        results_df["energy_temperature"] = resolved_temperature

    if args.embeddings:
        true_cbl = results_df["ground_truth"] != "NON-CBL"
        correct_serotype = results_df["pred_argmax"] == results_df["ground_truth"]
        correct_serotype = correct_serotype | ~true_cbl  # Ignore serotype accuracy for non-CBL samples
        correct_cbl = results_df["is_cbl"] == true_cbl
        true_pos = (correct_serotype & correct_cbl).mean()
        print(f"Serotype accuracy: {correct_serotype.mean():.4f}")
        print(f"CBL accuracy: {correct_cbl.mean():.4f}")
        print(f"True positive rate: {true_pos:.4f}")
    results_df \
        .drop(columns=["embedding", "serotype_logits"]) \
        .to_csv(os.path.join(args.output_dir, "query_results.csv"), index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Novel detection script.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output files.")
    parser.add_argument("--query", type=str, help="Path to the query FASTA file.")
    parser.add_argument("--embeddings", type=str, help="Path to the embeddings file.")
    parser.add_argument("--labels", type=str, help="Path to the labels file.")  # TODO should be a subset
    parser.add_argument("--distributions", type=str, default="distributions_params.json", help="Path to the distributions parameters file.")
    parser.add_argument("--min_serogroup_size", type=int, default=DEFAULT_MIN_SEROGROUP_SIZE,
        help="Minimum number of samples in a serogroup to be considered for novelty detection."
    )
    parser.add_argument("--thresholds", type=str, default=f"{THRESH_CPS},{THRESH_NONCPS},{NORM_NONCBL_PPF},{THRESH_BETA}",
        help="Comma-separated thresholds for novelty detection."  # TODO: explain the thresholds.
    )
    parser.add_argument("--model_params", type=str, default="{}",
                        help="JSON string of model parameters")
    parser.add_argument("--base_model", type=str, default=DEFAULT_MODEL,
                        help="Name of the base transformer model to use for initial inference.")
    parser.add_argument("--head_model", type=str, default="transformer_lr_classifier",
                        help="Name of the head model to use.")
    # New args for energy-based OOD
    parser.add_argument("--energy_temperature", type=float, default=1.0,
                        help="Temperature T for energy computation")
    parser.add_argument("--energy_percentile", type=float, default=99.0,
                        help="Percentile (e.g., 99) over ID energies to set tau_serotype")
    parser.add_argument("--id_energies_csv", type=str, default=None,
                        help="CSV with ID serotype energies (columns should include energy_serotype)")
    parser.add_argument("--energy_thresholds_json", type=str, default=None,
                        help="JSON file containing precomputed tau_serotype and temperature")

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

    if not args.query and not args.embeddings:
        parser.error("Either --query or --embeddings must be provided.")
    if args.query and args.embeddings:
        parser.error("Only one of --query or --embeddings can be provided.")
    if args.embeddings and not args.labels:
        parser.error("If --embeddings is provided, --labels must also be specified.")
    # In query mode, require source for tau
    if args.query and not (args.id_energies_csv or args.energy_thresholds_json):
        parser.error("In query mode, provide --id_energies_csv or --energy_thresholds_json for OOD thresholding.")

    return args

if __name__ == "__main__":
    main(parse_args())
