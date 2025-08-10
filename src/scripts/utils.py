import re
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

EPS = 1e-9

# A mapping of serotype to a more coarse label, consisting of
# similar groupings, i.e. genogroups or serogroups. We use it
# for contrastive training and visualization purposes.
SEROTYPE_GROUPS = {
    "6A": "6",
    "6B": "6",
    "6C": "6",
    "6D": "6",
    "6E(6B)": "6",
    "7B": "7B_7C_40",
    "7C": "7B_7C_40",
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
    "22A": "22",
    "22F": "22",
    "23A": "23",
    "23B": "23",
    "23B1": "23",
    "23F": "23",
    "24": "24",
    "24A": "24",
    "24F": "24",
    "25A": "25A_25F_38",
    "25F": "25A_25F_38",
    "28A": "28",
    "28F": "28",
    "33A": "33A_33F_37",
    "33A/33F": "33A_33F_37",
    "33B": "33B_33D",
    "33D": "33B_33D",
    "33F": "33A_33F_37",
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
    "46": "12_44_46"
}
WHITELIST = ["NON-CBL"]

def map_serotype_to_group(serotype):
    """ Map serotype to a more coarse label by
    looking it up in the serogroup/genogroups data. """
    if not isinstance(serotype, str):
        raise ValueError("Serotype must be a string.")
    
    if serotype in WHITELIST:
        return serotype
    if serotype in SEROTYPE_GROUPS:
        return SEROTYPE_GROUPS[serotype]

    return serotype


def extract_serogroup(serotype):
    """ Map serotype to a more coarse label by
    extracting the number from the serotype string. """
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

    diag_mask = torch.eye(N, dtype=torch.bool, device=device)  # Exclude diagonal from denominator

    exp_logits = torch.exp(logits)
    pos_exp = exp_logits * positives_mask
    numerator = pos_exp.sum(dim=1)  # (N,)

    den_exp = exp_logits * ~diag_mask
    denominator = den_exp.sum(dim=1)

    loss_terms = -torch.log((numerator + EPS) / (denominator + EPS))
    loss = loss_terms.mean()
    return loss


def hierarchical_contrastive_loss(z, labels, temperature, weight_fine=1.0, weight_coarse=0.5):
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
            if i == j: continue  # Exclude diagonal
            if coarse_labels[i] == coarse_labels[j]:
                if fine_labels[i] == fine_labels[j]:
                    weight_matrix[i, j] = weight_fine  # Strong positive
                else:
                    weight_matrix[i, j] = weight_coarse  # Partial positive, e.g., 15A vs 15B

    # An InfoNCE-like approach, but we sum up weighted positives in the numerator:
    #    Numerator = sum_{j} [ W[i, j] * exp(logits[i, j]) ]
    #    Denominator = sum_{k != i} [ exp(logits[i, k]) ]
    #    Then L_i = - log( ( numerator ) / ( denominator ) ), and final L = mean(L_i).

    diag_mask = torch.eye(N, dtype=torch.bool, device=device)  # Exclude diagonal from denominator

    exp_logits = torch.exp(logits)
    den_exp = exp_logits * ~diag_mask
    denominator = den_exp.sum(dim=1)  # shape (N,)

    num_exp = exp_logits * weight_matrix
    numerator = num_exp.sum(dim=1)

    loss_terms = -torch.log((numerator + EPS) / (denominator + EPS))
    loss = loss_terms.mean()
    return loss


def collate_fn(batch):
    embeddings = [item['embedding'] for item in batch]  # [(L_i, D), ...]
    serotypes = [item['serotype'] for item in batch]
    is_capsule = torch.tensor([item['is_capsule'] for item in batch], dtype=torch.long)

    padded_embeddings = pad_sequence(embeddings, batch_first=True)  # shape [B, L_max, D]

    return {
        'sample_id': [item['sample_id'] for item in batch],  # list[str]
        'embedding': padded_embeddings,   # tensor [B, L_max, D]
        'serotype': serotypes,            # list[str]
        'is_capsule': is_capsule          # tensor [B]
    }

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

