import re
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from typing import Tuple
import pandas as pd

from .consts import DEFAULT_SEP, CONTIG_SEP, DEFAULT_MISSING_LABEL, SEROGROUP_LABELS
from .logging_config import get_logger

logger = get_logger(__name__)
EPS = 1e-9

# A mapping of serotype to a more coarse label, consisting of
# CPS-locus-based genogroups.  References:
#   - Epping et al. 2018 (SeroBA), Table S1
#   - Kapatai et al. 2022 (PneumoCaT2), Table 1
#   - Bentley et al. 2006
# We use this for contrastive training and visualization purposes.
SEROTYPE_GROUPS = {
    "6A": "6",
    "6B": "6",
    "6C": "6",
    "6D": "6",
    "6E(6B)": "6",
    "7B": "7B_7C_40",
    "7C": "7B_7C_40",
    "7A": "7A_7F",
    "7F": "7A_7F",
    "9A": "9",
    "9L": "9",
    "9N": "9",
    "9V": "9",
    "10A": "10",
    "10B": "10",
    "10C": "10",
    "10F": "10",
    "10X": "33G",
    "11A": "11",
    "11B": "11",
    "11C": "11",
    "11D": "11",
    "11E": "11",
    "12A": "12_44_46",
    "12B": "12_44_46",
    "12F": "12_44_46",
    "15A": "15",
    "15B": "15",
    "15C": "15",
    "15B/15C": "15",
    "15F": "15",
    "18A": "18",
    "18B": "18",
    "18C": "18",
    "18F": "18",
    "19A": "19A",
    "19B": "19B_19C",
    "19F": "19F",
    "20": "20",
    "20A": "20",
    "20B": "20",
    "20C": "20",
    "22A": "22",
    "22F": "22",
    "23A": "23",
    "23B": "23",
    "23B1": "23",
    "23F": "23",
    "24A": "Serogroup 24",
    "24B": "Serogroup 24",
    "24F": "Serogroup 24",
    "25A": "25A_25F_38",
    "25F": "25A_25F_38",
    "28A": "28",
    "28F": "28",
    "32F": "32",
    "33A": "33A_33F_37",
    "33A/33F": "33A_33F_37",
    "33B": "33B_33D",
    "33D": "33B_33D",
    "33F": "33A_33F_37",
    "35A/42": "35A_35C_42",
    "35A": "35A_35C_42",
    "35B": "35B_35D",
    "35B/35D": "35B_35D",
    "35C": "35A_35C_42",
    "35D": "35B_35D",
    "37": "33A_33F_37",
    "38": "25A_25F_38",
    "39X": "10D",
    "40": "7B_7C_40",
    "42": "35A_35C_42",
    "46": "12_44_46",
    # Explicit entries for serogroup-level labels
    "24": "Serogroup 24",
    "Serogroup 24": "Serogroup 24",
    "33": "Serogroup 33",
    "Serogroup 33": "Serogroup 33",
}
WHITELIST = ["NON-CBL"]


def classify_label_type(label: str) -> str:
    """Classify a serotype label as 'serotype', 'serogroup_only', or 'compound'.

    - 'serogroup_only': The label is in SEROGROUP_LABELS (e.g. "Serogroup 24").
      These samples should NOT contribute to the serotype CE loss.
    - 'compound': The label contains '/' and is a key in SEROTYPE_GROUPS
      (e.g. "15B/15C", "33A/33F"). Cannot be resolved to a single serotype.
      These should also NOT contribute to the serotype CE loss.
    - 'serotype': A resolved, unambiguous serotype (e.g. "6A", "19F").
    """
    if label in SEROGROUP_LABELS:
        return "serogroup_only"
    if "/" in label:
        return "compound"
    return "serotype"


def map_serotype_to_group(serotype):
    """Map serotype to a more coarse label by
    looking it up in the serogroup/genogroups data."""
    if not isinstance(serotype, str):
        raise ValueError(
            "Serotype must be a string. Got: {} ({})".format(serotype, type(serotype))
        )

    if serotype in WHITELIST:
        return serotype
    if serotype in SEROTYPE_GROUPS:
        return SEROTYPE_GROUPS[serotype]

    return serotype


def extract_serogroup(serotype):
    """Map serotype to a more coarse label by
    extracting the number from the serotype string."""
    if isinstance(serotype, str):
        match = re.search(r"\d+", serotype)
        if match:
            return str(match.group())
    return serotype


def supervised_contrastive_loss(z, labels, temperature):
    """
    Supervised Contrastive Loss:
      - For each anchor i, all samples j with the same label are positives.
      - Different labels => negatives.
      - i != j (exclude diagonal).
    """
    device, N = z.device, z.shape[0]
    z = nn.functional.normalize(z, dim=1)  # Normalize embeddings for stable similarity
    logits = z @ z.t() / temperature  # shape (N, N)

    # positives_mask = torch.tensor([[(labels[i] == labels[j]) and (i != j) for j in range(N)] for i in range(N)], device=device)
    positives_mask = torch.zeros((N, N), dtype=torch.bool, device=device)
    for i in range(N):
        for j in range(N):
            if i != j and labels[i] == labels[j]:
                positives_mask[i, j] = True

    diag_mask = torch.eye(
        N, dtype=torch.bool, device=device
    )  # Exclude diagonal from denominator

    exp_logits = torch.exp(logits)
    pos_exp = exp_logits * positives_mask
    numerator = pos_exp.sum(dim=1)  # (N,)

    den_exp = exp_logits * ~diag_mask
    denominator = den_exp.sum(dim=1)

    loss_terms = -torch.log((numerator + EPS) / (denominator + EPS))
    loss = loss_terms.mean()
    return loss


def hierarchical_contrastive_loss(
    z, labels, temperature, weight_fine=1.0, weight_coarse=0.5
):
    """
    Hierarchical contrastive loss that assigns different weights to pairs that share:
      (a) the same fine label (strong positive),
      (b) the same coarse label but different fine label (partial positive),
      (c) different coarse label (negative).
    """
    device, N = z.device, z.shape[0]
    coarse_labels, fine_labels = zip(*labels)

    z = nn.functional.normalize(z, dim=1)
    logits = z @ z.t() / temperature

    weight_matrix = torch.zeros((N, N), dtype=torch.float, device=device)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue  # Exclude diagonal
            if coarse_labels[i] == coarse_labels[j]:
                if fine_labels[i] == fine_labels[j]:
                    weight_matrix[i, j] = weight_fine  # Strong positive
                else:
                    weight_matrix[i, j] = (
                        weight_coarse  # Partial positive, e.g., 15A vs 15B
                    )

    # An InfoNCE-like approach, but we sum up weighted positives in the numerator:
    #    Numerator = sum_{j} [ W[i, j] * exp(logits[i, j]) ]
    #    Denominator = sum_{k != i} [ exp(logits[i, k]) ]
    #    Then L_i = - log( ( numerator ) / ( denominator ) ), and final L = mean(L_i).

    diag_mask = torch.eye(
        N, dtype=torch.bool, device=device
    )  # Exclude diagonal from denominator

    exp_logits = torch.exp(logits)
    den_exp = exp_logits * ~diag_mask
    denominator = den_exp.sum(dim=1)  # shape (N,)

    num_exp = exp_logits * weight_matrix
    numerator = num_exp.sum(dim=1)

    loss_terms = -torch.log((numerator + EPS) / (denominator + EPS))
    loss = loss_terms.mean()
    return loss


def collate_fn(batch):
    embeddings = [item["embedding"] for item in batch]  # [(L_i, D), ...]
    serotypes = [item["serotype"] for item in batch]
    is_capsule = torch.tensor([item["is_capsule"] for item in batch], dtype=torch.long)
    serotype_known = torch.tensor(
        [item.get("serotype_known", True) for item in batch], dtype=torch.bool
    )

    padded_embeddings = pad_sequence(
        embeddings, batch_first=True
    )  # shape [B, L_max, D]

    return {
        "sample_id": [item["sample_id"] for item in batch],  # list[str]
        "embedding": padded_embeddings,  # tensor [B, L_max, D]
        "serotype": serotypes,  # list[str]
        "is_capsule": is_capsule,  # tensor [B]
        "serotype_known": serotype_known,  # tensor [B] (bool)
    }


def chunk_sequence(seq, chunk_size=512, stride=256):
    return [
        seq[i : i + chunk_size] for i in range(0, len(seq) - chunk_size + 1, stride)
    ]


def embed_chunks(chunks, tokenizer, model, device, max_length):
    # Ensure model is in eval mode for consistent embeddings
    model.eval()
    inputs = tokenizer(
        chunks,
        return_tensors="pt",
        # padding="max_length",  # TODO
        # truncation=True,  # TODO
        # max_length=max_length  # TODO
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # (B, T, D)
        pooled = last_hidden.mean(dim=1)  # (B, D)
    return pooled.cpu()


def get_sample_id(df: pd.DataFrame, sep=CONTIG_SEP) -> pd.Series:
    return df.index + sep + df["Contig_ID"].astype(str)


def load_data(
    embeddings_path: str,
    labels_path: str,
    sep: str = DEFAULT_SEP,
    missing_label: str = DEFAULT_MISSING_LABEL,
) -> Tuple[np.ndarray, pd.DataFrame]:
    X = np.load(embeddings_path, allow_pickle=True)  # shape: (N, L, D)
    labels_df = pd.read_csv(
        labels_path, index_col=0, sep="\t" if labels_path.endswith(".tsv") else ","
    )
    labels_df["Serotype"] = labels_df["Serotype"].fillna(
        missing_label
    )  # TODO should be empty already
    labels_df = labels_df[labels_df["Serotype"] != missing_label]

    keys = (
        labels_df["Is_capsule"].map(lambda x: "cbl" if x else "non-cbl")
        + sep
        + get_sample_id(labels_df)
    )
    X_filtered = np.stack([X[key] for key in keys])
    logger.info("Loaded %d embeddings for capsulated samples", len(X_filtered))
    return X_filtered, labels_df
