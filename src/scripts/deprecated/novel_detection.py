import argparse
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from typing import List, Tuple
from scipy.stats import beta, norm
from sklearn.metrics.pairwise import cosine_distances
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from models import TransformerContrastiveHead
from consts import (
    DEFAULT_MIN_SEROGROUP_SIZE, DEFAULT_MODEL, DEFAULT_CHUNK_SIZE,
    DEFAULT_STRIDE_RATIO, DEFAULT_MAX_LEN, DEFAULT_EMBEDDING_DIM,
    DEFAULT_OUTPUT_DIM, DEFAULT_NHEAD, DEFAULT_NUM_LAYERS,
    DEFAULT_SEP, DEFAULT_LABEL_COLUMN, DEFAULT_MISSING_LABEL,
)

EPS = 1e-6
THRESH_CPS = 0.9  # TODO clean up and document and verify and what the fuck
THRESH_NONCPS = 0.1
NORM_NONCBL_PPF = 0.95
THRESH_BETA = 0.98


def transformer_embedding(  # TODO batch this 
    tokenizer,
    nt_model,
    contrastive_model: TransformerContrastiveHead,
    sequences: List[str],
    device: str = "cuda",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    stride_ratio: float = DEFAULT_STRIDE_RATIO,
    max_length: int = DEFAULT_MAX_LEN,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a list of sequences, chunk and embed them using a Nucleotide Transformer,
    then feed through the trained TransformerContrastiveHead to get final embeddings.
    Returns: np.ndarray of shape (len(sequences), output_dim)
    """
    max_length = tokenizer.model_max_length
    chunk_size = min(chunk_size, max_length)
    stride = int(chunk_size * stride_ratio)

    all_logits = []
    all_embeddings = []
    for seq in sequences:
        # Chunk the sequence
        chunks = [seq[i:i + chunk_size] for i in range(0, len(seq) - chunk_size + 1, stride)]
        if not chunks:
            continue

        # Embed each chunk
        inputs = tokenizer(
            chunks,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = nt_model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]  # (L, T, D)
            pooled = last_hidden.mean(dim=1)         # (L, D)

        # Feed through contrastive model
        with torch.no_grad():
            logits, embedding = contrastive_model(pooled.unsqueeze(0))  # (1, L, D) -> (1, output_dim)
            embedding = embedding.squeeze(0).cpu().numpy()  # (output_dim,)
        all_logits.append(logits.cpu().numpy())
        all_embeddings.append(embedding)

    return np.stack(all_logits), np.stack(all_embeddings)
    

def fit_distributions(labels: pd.DataFrame, distances: np.ndarray, min_serogroup_size: int) -> Tuple[dict, dict]:
    """
    Fit normal and beta distributions for serogroups based on pairwise distances.
    """
    labels_cp = labels.reset_index().iloc[:, 1]  # Ensure labels are indexed from 0
    serogroups = labels_cp.unique()
    beta_params = {}
    for serogroup in serogroups:
        indices = labels_cp[labels_cp == serogroup].index
        if len(indices) < min_serogroup_size:
            print(f"Skipping serogroup {serogroup} with size {len(indices)} < {min_serogroup_size}.")
            continue
        sub_distances = distances[np.ix_(indices, indices)][np.triu_indices(len(indices), k=1)]
        sub_distances = sub_distances[sub_distances > EPS]
        a, b, loc, scale = beta.fit(sub_distances, floc=0, fscale=2)  # 2 because of cosine_distance
        beta_params[serogroup] = {"a": float(a), "b": float(b)}  # TODO Should I store "loc": loc, "scale": scale?

    # Could either fit a normal or a GMM with 2 components
    normal_params = norm.fit(distances.flatten())
    normal_params = {
        "mu": float(normal_params[0]),
        "sigma": float(normal_params[1])
    }
    return beta_params, normal_params


def generate_verdict(results: pd.DataFrame, beta_params: dict, normal_params: dict, output_dir: str):
    out_path = os.path.join(output_dir, "novel_detection_report.txt")
    # TODO
    for query in results["Query"].unique():
        query_results = results[results["Query"] == query]
        for _, row in query_results.iterrows():
            serogroup = row["Serogroup"]
            if serogroup != "All":
                params = beta_params[serogroup]
                threshold = beta.ppf(row["Threshold"], params["a"], params["b"])
                if row["query_distance_mean"] < threshold:
                    row["Novel"] = "Novel"
                else:
                    row["Novel"] = "Known"
            else:
                threshold = norm.ppf(row["Threshold"], loc=normal_params["mu"], scale=normal_params["sigma"])
                if row["query_distance_mean"] < threshold:
                    row["Novel"] = "Novel"
                else:
                    row["Novel"] = "Known"
    with open(out_path, "w") as f:
        f.write("Individual statistics:\n")
        f.write("\nVerdict:\n")
        results.to_csv(f, sep="\t", index=False)
    # results.groupby("Serogroup")["Novel"].value_counts().unstack(fill_value=0)


def main(args):
    ### TODO introduce params
    nt_model_name = DEFAULT_MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sep = DEFAULT_SEP
    chunk_size = DEFAULT_CHUNK_SIZE
    stride_ratio = DEFAULT_STRIDE_RATIO
    max_length = DEFAULT_MAX_LEN
    embedding_dim = DEFAULT_EMBEDDING_DIM
    output_dim = DEFAULT_OUTPUT_DIM
    nhead = DEFAULT_NHEAD
    num_layers = DEFAULT_NUM_LAYERS
    label_column = DEFAULT_LABEL_COLUMN
    missing_label = DEFAULT_MISSING_LABEL
    min_count = DEFAULT_MIN_SEROGROUP_SIZE
    
    thresholds = list(map(float, map(str.strip, args.thresholds.split(","))))
    assert len(thresholds) == 4, "Four thresholds are required."
    assert all(0 < t < 1 for t in thresholds), "Thresholds must be between 0 and 1."
    thresh_cps, thresh_noncps, norm_noncbl_ppf, thresh_beta = thresholds
    ### 
    print("Loading embeddings and labels...")
    embeddings = np.load(args.embeddings)
    labels = pd.read_csv(args.labels, index_col=0, sep="\t")
    
    labels[label_column] = labels[label_column].fillna(missing_label)
    known_indices = labels[label_column] != missing_label
    ### Not gonna drop I guess. Just skipping the beta fitting.
    # underrep_labels = labels[label_column].value_counts()[labels[label_column].value_counts() < min_count].index
    # if underrep_labels.any():
    #     print(f"Dropping serotypes with less than {min_count} samples:", *underrep_labels.to_list())
    #     known_indices &= ~labels[label_column].isin(underrep_labels)
    labels = labels[known_indices]
    labels = labels[labels[label_column] != "2"]  # TODO fix by feeding a subset of labels to this script in a standalone workflow
    
    is_emb_npz = isinstance(embeddings, np.lib.npyio.NpzFile)
    if is_emb_npz:
        cbl_prefix = lambda k: f"cbl{sep}{k}"
        embeddings = np.array([embeddings[cbl_prefix(key)] for key in labels.index if cbl_prefix(key) in embeddings.keys()])
        # TODO Option to show non-cbl too
    assert embeddings.shape[0] == len(labels), "Number of embeddings does not match number of labels."

    print("Calculating pairwise distances...")
    dist_matrix = cosine_distances(embeddings)  # Should have calculated here instead

    print("Parsing thresholds and parameters...")
    params_path = os.path.join(args.output_dir, args.distributions)
    if os.path.exists(params_path):
        print("\tLoading distribution parameters...")
        params_data = json.load(open(params_path))
        beta_params, normal_params = params_data["beta"], params_data["normal"]
    else:
        print("\tParameters not found. Fitting beta distributions...")
        beta_params, normal_params = fit_distributions(labels[[label_column]], dist_matrix, min_count)
        print("\tSaving distribution parameters...")
        json.dump(
            {
                "beta": beta_params,
                "normal": normal_params
            },
            open(params_path, "w"),
        )

    print("Loading the transformer model and contrastive head...")
    tokenizer = AutoTokenizer.from_pretrained(nt_model_name)
    nt_model = AutoModelForMaskedLM.from_pretrained(nt_model_name).to(device)
    
    contrastive_model = TransformerContrastiveHead(
        input_dim=embedding_dim,
        output_dim=output_dim,
        nhead=nhead,
        num_layers=num_layers
    ).to(device)
    model_path = os.path.join(args.output_dir, "contrastive_model.pth")
    contrastive_model.load_state_dict(torch.load(model_path, map_location=device))
    contrastive_model.eval()

    results = []
    print("Processing queries...")
    total = sum(1 for _ in SeqIO.parse(args.query, "fasta"))
    for record in tqdm(SeqIO.parse(args.query, "fasta"), total=total):
        print(f"Processing query: {record.id}...")
        print("\t- Embedding the query sequence...")
        query_logits, query_embedding = transformer_embedding(
            tokenizer=tokenizer,
            nt_model=nt_model,
            contrastive_model=contrastive_model,
            sequences=[str(record.seq)],
            device=device,
            chunk_size=chunk_size,
            stride_ratio=stride_ratio,
            max_length=max_length,
        )
        query_logits = torch.sigmoid(torch.tensor(query_logits)).numpy().flatten()
        argmax_idx = np.argmax(query_logits)
        prediction = "CBL" if argmax_idx == 1 else "Non-CBL"
        print(f"Classification for {record.id}: {prediction}, confidence: {query_logits.max():.4f}")
        if query_logits[argmax_idx] < thresh_cps:
            print(f"The model is not confident about Query {record.id} being a CPS. Confidence: {query_logits[argmax_idx]:.4f}.")
        if query_logits[0] > thresh_cps:
            print(f"Query {record.id} is classified as non-CBL with confidence {query_logits[0]:.4f}.")
            results.append({
                "Query": record.id,
                "Description": record.description,
                "Prediction": "Non-CBL",
                "Threshold": query_logits[0],
                "PPF": None,  # No PPF for non-CBL
                "query_distance_mean": None,  # Will be filled later
                "query_distance_std": None,  # Will be filled later
                "#Samples": None,  # TODO Will be filled later
            })
        elif query_logits[1] > thresh_cps:
            print(f"Query {record.id} is classified as CBL with confidence {query_logits[1]:.4f}.")
            
        query_distances = cosine_distances(query_embedding, embeddings).flatten()

        # Evaluate against each serogroup
        print("reached beta params")
        for serogroup, params in beta_params.items():
            ppf_threshold = beta.ppf(thresh_beta, params["a"], params["b"])
            results.append({
                "Query": record.id,
                "Prediction": serogroup,
                "Threshold": thresh_beta,
                "PPF": ppf_threshold,
                "query_distance_mean": query_distances[labels[label_column] == serogroup].mean(),
                "query_distance_std": query_distances[labels[label_column] == serogroup].std(),
                "#Samples": len(labels[labels[label_column] == serogroup]),
            })
        # Evaluate against everything
        ppf_threshold = norm.ppf(norm_noncbl_ppf, loc=normal_params["mu"], scale=normal_params["sigma"])
        results.append({
            "Query": record.id,
            "Prediction": "All",
            "Threshold": norm_noncbl_ppf,
            "PPF": ppf_threshold,
            "query_distance_mean": query_distances.mean(),
            "query_distance_std": query_distances.std(),
            "#Samples": len(labels),
        })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.output_dir, "novel_detection_results.csv"), index=False)
    print("Saving results...")
    print(results_df)
    # TODO ROC curve, AUC, etc.
    # generate_verdict(results_df, beta_params, normal_params, args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Novel detection script.")
    parser.add_argument("--query", type=str, required=True, help="Path to the query FASTA file.")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to the embeddings file.")
    parser.add_argument("--labels", type=str, required=True, help="Path to the labels file.")  # TODO should be a subset
    parser.add_argument("--distances", type=str, required=True, help="Path to the distances file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output files.")
    parser.add_argument("--distributions", type=str, default="distributions_params.json", help="Path to the distributions parameters file.")
    parser.add_argument("--min_serogroup_size", type=int, default=DEFAULT_MIN_SEROGROUP_SIZE,
        help="Minimum number of samples in a serogroup to be considered for novelty detection."
    )
    parser.add_argument("--thresholds", type=str, default=f"{THRESH_CPS},{THRESH_NONCPS},{NORM_NONCBL_PPF},{THRESH_BETA}",
        help="Comma-separated thresholds for novelty detection."  # TODO: explain the thresholds.
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
