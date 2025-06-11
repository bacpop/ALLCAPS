import argparse
import os
import json
import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy.stats import beta, norm
from sklearn.metrics.pairwise import cosine_distances

MIN_SEROGROUP_SIZE = 40  # Minimum number of samples in a serogroup to be considered for novelty detection

def transformer_embedding(sequence: str) -> np.ndarray:
    raise NotImplementedError("This function should be implemented to return the embedding of the sequence.")

def fit_distributions(labels: pd.Series, distances: np.ndarray) -> tuple:
    """
    Fit normal and beta distributions for serogroups based on pairwise distances.
    """
    serogroups = labels.unique()
    beta_params = {}
    for serogroup in serogroups:
        indices = labels[labels == serogroup].index
        sub_distances = distances[np.ix_(indices, indices)][np.triu_indices(len(indices), k=1)]
        a, b, loc, scale = beta.fit(sub_distances, floc=0, fscale=1)
        beta_params[serogroup] = {"a": a, "b": b}  # Should I store "loc": loc, "scale": scale?
        
    # Could either fit a normal or a GMM with 2 components
    normal_params = norm.fit(distances.flatten())
    normal_params = {
        "mu": normal_params[0],
        "sigma": normal_params[1]
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
    print("Loading embeddings and labels...")
    embeddings = np.load(args.embeddings)
    labels = pd.read_csv(args.labels, index_col=0)["Serotype"]

    print("Loading the distance matrix...")
    dist_data = np.load(args.distances)
    dist_matrix = np.zeros((dist_data["size"], dist_data["size"]))
    dist_matrix[np.triu_indices_from(dist_matrix, k=1)] = dist_data["distances"]
    dist_matrix += dist_matrix.T

    print("Parsing thresholds and parameters...")
    params_path = os.path.join(args.output_dir, args.distributions)
    if os.path.exists(params_path):
        print("\tLoading distribution parameters...")
        params_data = json.load(open(args.distributions))
        beta_params, normal_params = params_data["beta"], params_data["normal"]
    else:
        print("\tParameters not found. Fitting beta distributions...")
        beta_params, normal_params = fit_distributions(labels, dist_matrix)
        print("\tSaving distribution parameters...")
        json.dump(
            {
                "beta": beta_params,
                "normal": normal_params
            },
            open(args.distributions, "w"),
        )

    thresholds = list(map(float, map(str.strip, args.thresholds.split(","))))
    assert len(thresholds) == 4, "Four thresholds are required."
    assert all(0 < t < 1 for t in thresholds), "Thresholds must be between 0 and 1."
    THRESH_CPS, THRESH_NONCPS, NORM_NONCBL_PPF, THRESH_BETA = thresholds

    results = []
    for record in SeqIO.parse(args.query, "fasta"):
        print(f"Processing query: {record.id}...")
        query_embedding = transformer_embedding(str(record.seq))
        query_distances = cosine_distances(query_embedding, embeddings).flatten()

        # Evaluate against each serogroup
        for serogroup, params in beta_params.items():
            ppf_threshold = beta.ppf(THRESH_BETA, params["a"], params["b"])
            results.append({
                "Query": record.id,
                "Serogroup": serogroup,
                "Threshold": THRESH_BETA,
                "PPF": ppf_threshold,
                "query_distance_min": query_distances[labels == serogroup].min(),  # Use min or mean?
                "query_distance_mean": query_distances[labels == serogroup].mean(),
                "#Samples": len(labels[labels == serogroup]),
            })
        # Evaluate against everything
        ppf_threshold = norm.ppf(NORM_NONCBL_PPF, loc=normal_params["mu"], scale=normal_params["sigma"])
        results.append({
            "Query": record.id,
            "Serogroup": "All",
            "Threshold": NORM_NONCBL_PPF,
            "PPF": ppf_threshold,
            "query_distance_min": query_distances.min(),
            "query_distance_mean": query_distances.mean(),
            "#Samples": len(labels),
        })

    # Save results
    results_df = pd.DataFrame(results)
    print("Saving results...")
    generate_verdict(results_df, beta_params, normal_params, args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Novel detection script.")
    parser.add_argument("--query", type=str, required=True, help="Path to the query FASTA file.")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to the embeddings file.")
    parser.add_argument("--labels", type=str, required=True, help="Path to the labels file.")
    parser.add_argument("--distances", type=str, required=True, help="Path to the distances file.")
    parser.add_argument("--distributions", type=str, default="distributions_params.json", help="Path to the distributions parameters file.")
    parser.add_argument("--thresholds", type=str, default="0.05,0.1,0.95,0.98",
        help="Comma-separated thresholds for novelty detection."
            " TODO: explain the thresholds."
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the output files.")
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())