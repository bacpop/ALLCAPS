import os
import json
import wandb
import argparse
from tqdm import tqdm
from functools import partial

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from ..models import TransformerTriHeadLR, DatasetRegistry
from ..utils import (
    supervised_contrastive_loss,
    hierarchical_contrastive_loss,
    map_serotype_to_group,
    collate_fn,
    classify_label_type,
)
from ..data_augmentation import EmbeddingAugmentor, AugmentedChunkedDataset
from ..logging_config import get_logger

from ..consts import (
    RND_STATE,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_KFOLDS,
    DEFAULT_TEMPERATURE,
    DEFAULT_WEIGHT_FINE,
    DEFAULT_WEIGHT_COARSE,
    DEFAULT_NUM_LAYERS,
    DEFAULT_NHEAD,
    DEFAULT_OUTPUT_DIM,
    DEFAULT_EMBEDDING_DIM,
    CONTIG_SEP,
    DEFAULT_MISSING_LABEL,
    DEFAULT_LABEL_COLUMN,
    DEFAULT_EARLY_STOPPING,
    DEFAULT_CONTRASTIVE_LOSS_RATIO,
    MIN_SEROTYPE_COUNT,
)

from collections import Counter

logger = get_logger(__name__)

EPS = 1e-9
ALPHA_SERO = 2
ALPHA_GENO = 1
WANDB_PROJECT_NAME = "logistic-trihead-augment"


def compute_class_weights(labels, label_to_idx, device):
    """Compute balanced class weights: w_c = N / (C * n_c).

    Parameters
    ----------
    labels : array-like
        Per-sample string labels (only the training split).
    label_to_idx : dict
        Mapping from label string to class index.
    device : torch.device

    Returns
    -------
    torch.Tensor of shape (num_classes,) on *device*.
    """
    counts = Counter(labels)
    n_total = len(labels)
    n_classes = len(label_to_idx)
    weights = torch.ones(n_classes, dtype=torch.float32)
    for label, idx in label_to_idx.items():
        n_c = counts.get(label, 0)
        weights[idx] = n_total / (n_classes * n_c) if n_c > 0 else 1.0
    return weights.to(device)


DEFAULT_DATASET_OBJECT = "contrastive_chunked"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding_dir",
        required=True,
        help="Directory containing chunked embeddings in npy format.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument(
        "--model_params",
        type=str,
        default="{}",
        help="JSON string of model parameters (output_dim, num_layers, nhead, alpha, temperature, etc.)",
    )
    parser.add_argument("--labeled_only", action="store_true")
    parser.add_argument(
        "--skip_labels",
        type=str,
        default="",
        help="Comma-separated list of labels to skip in training.",
    )
    parser.add_argument(
        "--hierarchical_loss",
        action="store_true",
        help="Use weighted (coarse, fine) labels for training.",
    )
    parser.add_argument("--early_stopping", type=int, default=DEFAULT_EARLY_STOPPING)
    # Embedding-level augmentation (applied on-the-fly during training only)
    parser.add_argument(
        "--aug_noise_std",
        type=float,
        default=0.0,
        help="Gaussian noise std for embedding augmentation (0 = disabled).",
    )
    parser.add_argument(
        "--aug_chunk_dropout",
        type=float,
        default=0.0,
        help="Chunk dropout probability for embedding augmentation (0 = disabled).",
    )
    parser.add_argument(
        "--aug_spec_freq",
        type=float,
        default=0.0,
        help="SpecAugment frequency masking probability (0 = disabled).",
    )
    parser.add_argument(
        "--aug_spec_width",
        type=int,
        default=16,
        help="SpecAugment max mask width in feature dimensions.",
    )
    parser.add_argument(
        "--aug_n_views",
        type=int,
        default=1,
        help="Number of augmented views per sample (1 = original only, 2+ = contrastive views).",
    )
    args = parser.parse_args()

    try:
        args.model_params = json.loads(args.model_params)
        if not isinstance(args.model_params, dict):
            logger.warning("Model parameters should be a JSON object.")
            args.model_params = {}
    except json.JSONDecodeError:
        logger.error("Error parsing model parameters JSON string.")
        args.model_params = {}
    finally:
        logger.info("Model parameters: %s", args.model_params)

    try:
        args.skip_labels = [
            label.strip() for label in args.skip_labels.split(",") if label.strip()
        ]
    except ValueError:
        logger.warning(
            "Error parsing skip_labels. It should be a comma-separated list of labels. Proceeding with no skips."
        )
        args.skip_labels = []

    return args


def train_one_epoch(
    model,
    loader,
    optimizer,
    ce_loss_fn,
    serotype_loss_fn,
    genogroup_loss_fn,
    contrastive_loss_fn,
    alpha,
    temperature,
    serotype_to_idx,
    genogroup_to_idx,
):
    model.train()
    total_loss, ce_loss, serotype_loss, genogroup_loss, contrastive_loss = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    for batch in loader:
        capsule_label = batch["is_capsule"].cuda()
        serotype_label = batch["serotype"]
        serotype_known_batch = batch["serotype_known"].cuda()  # bool tensor
        capsule_mask = capsule_label == 1
        # Serotype loss: only capsulated AND resolved-serotype samples
        serotype_mask = capsule_mask & serotype_known_batch

        # Unpack labels
        coarse_labels = []
        fine_labels = []
        for lbl in serotype_label:
            if isinstance(lbl, (list, tuple, np.ndarray)) and len(lbl) >= 2:
                coarse_labels.append(lbl[0])
                fine_labels.append(lbl[-1])
            else:
                fine_labels.append(lbl)
                coarse_labels.append(map_serotype_to_group(lbl))

        cbl_logits, (serotype_logits, genogroup_logits), embeddings = model(
            batch["embedding"].cuda()
        )

        # CBL classification loss
        ce_loss_val = ce_loss_fn(cbl_logits, capsule_label)

        # Serotype classification loss (only for capsule + resolved-serotype samples)
        serotype_loss_val = torch.tensor(0.0, device=ce_loss_val.device)
        genogroup_loss_val = torch.tensor(0.0, device=ce_loss_val.device)
        if serotype_mask.sum() > 0:
            serotype_indices = torch.tensor(
                [
                    serotype_to_idx[fine_labels[i]]
                    for i in range(len(capsule_label))
                    if serotype_mask[i]
                ],
                device=ce_loss_val.device,
            )
            serotype_loss_val = serotype_loss_fn(
                serotype_logits[serotype_mask], serotype_indices
            )

        # Genogroup loss (all capsulated samples, including serogroup-only)
        if capsule_mask.sum() > 0:
            genogroup_indices = torch.tensor(
                [
                    genogroup_to_idx[coarse_labels[i]]
                    for i in range(len(capsule_label))
                    if capsule_mask[i]
                ],
                device=ce_loss_val.device,
            )
            genogroup_loss_val = genogroup_loss_fn(
                genogroup_logits[capsule_mask], genogroup_indices
            )

        # Contrastive loss (all capsulated samples)
        contrastive_loss_val = torch.tensor(0.0, device=ce_loss_val.device)
        if capsule_mask.sum() > 1:
            contrastive_loss_val = contrastive_loss_fn(
                embeddings[capsule_mask],
                [
                    serotype_label[i]
                    for i in range(len(capsule_label))
                    if capsule_mask[i]
                ],
                temperature,
            )

        loss = (
            ce_loss_val
            + ALPHA_SERO * serotype_loss_val
            + ALPHA_GENO * genogroup_loss_val
            + alpha * contrastive_loss_val
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        ce_loss += ce_loss_val.item()
        serotype_loss += serotype_loss_val.item()
        genogroup_loss += genogroup_loss_val.item()
        contrastive_loss += contrastive_loss_val.item()

    wandb.log(
        {
            "epoch_loss": total_loss / len(loader),
            "epoch_ce_loss": ce_loss / len(loader),
            "epoch_serotype_loss": serotype_loss / len(loader),
            "epoch_genogroup_loss": genogroup_loss / len(loader),
            "epoch_contrastive_loss": contrastive_loss / len(loader),
        }
    )
    return total_loss / len(loader)


def evaluate(
    model,
    loader,
    ce_loss_fn,
    serotype_loss_fn,
    genogroup_loss_fn,
    contrastive_loss_fn,
    alpha,
    temperature,
    serotype_to_idx,
    genogroup_to_idx,
):
    model.eval()
    total_loss = 0.0
    correct_cbl = 0
    correct_serotype = 0
    correct_genogroup = 0
    total_cbl = 0
    total_serotype = 0
    total_genogroup = 0
    with torch.no_grad():
        for batch in loader:
            capsule_label = batch["is_capsule"].cuda()
            serotype_label = batch["serotype"]
            serotype_known_batch = batch["serotype_known"].cuda()
            capsule_mask = capsule_label == 1
            serotype_mask = capsule_mask & serotype_known_batch

            coarse_labels = []
            fine_labels = []
            for lbl in serotype_label:
                if isinstance(lbl, (list, tuple, np.ndarray)) and len(lbl) >= 2:
                    coarse_labels.append(lbl[0])
                    fine_labels.append(lbl[-1])
                else:
                    fine_labels.append(lbl)
                    coarse_labels.append(map_serotype_to_group(lbl))

            cbl_logits, (serotype_logits, genogroup_logits), embeddings = model(
                batch["embedding"].cuda()
            )

            # CBL classification loss
            ce_loss_val = ce_loss_fn(cbl_logits, capsule_label)

            # Serotype classification loss (only for capsule + resolved-serotype samples)
            serotype_loss_val = torch.tensor(0.0, device=ce_loss_val.device)
            genogroup_loss_val = torch.tensor(0.0, device=ce_loss_val.device)
            if serotype_mask.sum() > 0:
                serotype_indices = torch.tensor(
                    [
                        serotype_to_idx[fine_labels[i]]
                        for i in range(len(capsule_label))
                        if serotype_mask[i]
                    ],
                    device=ce_loss_val.device,
                )
                serotype_loss_val = serotype_loss_fn(
                    serotype_logits[serotype_mask], serotype_indices
                )

            # Genogroup loss (all capsulated)
            if capsule_mask.sum() > 0:
                genogroup_indices = torch.tensor(
                    [
                        genogroup_to_idx[coarse_labels[i]]
                        for i in range(len(capsule_label))
                        if capsule_mask[i]
                    ],
                    device=ce_loss_val.device,
                )
                genogroup_loss_val = genogroup_loss_fn(
                    genogroup_logits[capsule_mask], genogroup_indices
                )

            # Contrastive loss (all capsulated)
            contrastive_loss_val = torch.tensor(0.0, device=ce_loss_val.device)
            if capsule_mask.sum() > 1:
                contrastive_loss_val = contrastive_loss_fn(
                    embeddings[capsule_mask],
                    [
                        serotype_label[i]
                        for i in range(len(capsule_label))
                        if capsule_mask[i]
                    ],
                    temperature,
                )

            loss = (
                ce_loss_val
                + ALPHA_SERO * serotype_loss_val
                + ALPHA_GENO * genogroup_loss_val
                + alpha * contrastive_loss_val
            )
            total_loss += loss.item()

            # CBL accuracy
            _, predicted_cbl = torch.max(cbl_logits, 1)
            correct_cbl += (predicted_cbl == capsule_label).sum().item()
            total_cbl += capsule_label.size(0)

            # Serotype accuracy (only for resolved-serotype samples)
            if serotype_mask.sum() > 0:
                _, predicted_serotype = torch.max(serotype_logits[serotype_mask], 1)
                serotype_indices = torch.tensor(
                    [
                        serotype_to_idx[fine_labels[i]]
                        for i in range(len(capsule_label))
                        if serotype_mask[i]
                    ],
                    device=ce_loss_val.device,
                )
                correct_serotype += (
                    (predicted_serotype == serotype_indices).sum().item()
                )
                total_serotype += serotype_indices.size(0)

            # Genogroup accuracy (all capsulated)
            if capsule_mask.sum() > 0:
                _, predicted_genogroup = torch.max(genogroup_logits[capsule_mask], 1)
                genogroup_indices = torch.tensor(
                    [
                        genogroup_to_idx[coarse_labels[i]]
                        for i in range(len(capsule_label))
                        if capsule_mask[i]
                    ],
                    device=ce_loss_val.device,
                )
                correct_genogroup += (
                    (predicted_genogroup == genogroup_indices).sum().item()
                )
                total_genogroup += genogroup_indices.size(0)

    cbl_accuracy = correct_cbl / total_cbl if total_cbl > 0 else 0.0
    serotype_accuracy = correct_serotype / total_serotype if total_serotype > 0 else 0.0
    genogroup_accuracy = (
        correct_genogroup / total_genogroup if total_genogroup > 0 else 0.0
    )

    wandb.log(
        {
            "test_loss": total_loss / len(loader),
            "cbl_accuracy": cbl_accuracy,
            "serotype_accuracy": serotype_accuracy,
            "genogroup_accuracy": genogroup_accuracy,
        }
    )
    return total_loss / len(loader), cbl_accuracy, serotype_accuracy, genogroup_accuracy


def main(args):
    device = args.device
    random_state = args.model_params.get("random_state", RND_STATE)
    k_folds = args.model_params.get("k_folds", DEFAULT_KFOLDS)
    temperature = args.model_params.get("temperature", DEFAULT_TEMPERATURE)
    weight_fine = args.model_params.get("weight_fine", DEFAULT_WEIGHT_FINE)
    weight_coarse = args.model_params.get("weight_coarse", DEFAULT_WEIGHT_COARSE)
    num_layers = args.model_params.get("num_layers", DEFAULT_NUM_LAYERS)
    nhead = args.model_params.get("nhead", DEFAULT_NHEAD)
    alpha = args.model_params.get("alpha", DEFAULT_CONTRASTIVE_LOSS_RATIO)
    output_dim = args.model_params.get("output_dim", DEFAULT_OUTPUT_DIM)
    embedding_dim = args.model_params.get("embedding_dim", DEFAULT_EMBEDDING_DIM)

    dataset_name = args.model_params.get("dataset_name", DEFAULT_DATASET_OBJECT)
    missing_label = args.model_params.get("missing_label", DEFAULT_MISSING_LABEL)
    label_column = args.model_params.get("label_column", DEFAULT_LABEL_COLUMN)

    # ── Build embedding augmentor (training-only) ──
    aug_enabled = (
        args.aug_noise_std > 0 or args.aug_chunk_dropout > 0 or args.aug_spec_freq > 0
    )
    embedding_augmentor = None
    if aug_enabled:
        embedding_augmentor = EmbeddingAugmentor(
            noise_std=args.aug_noise_std,
            chunk_dropout=args.aug_chunk_dropout,
            spec_augment_freq=args.aug_spec_freq,
            spec_augment_width=args.aug_spec_width,
        )
        logger.info(
            "Embedding augmentation enabled: noise_std=%s, chunk_dropout=%s, spec_freq=%s, spec_width=%s, n_views=%s",
            args.aug_noise_std,
            args.aug_chunk_dropout,
            args.aug_spec_freq,
            args.aug_spec_width,
            args.aug_n_views,
        )

    logger.info("Loading data...")
    labels = pd.read_csv(
        args.labels, index_col=0, sep="\t" if args.labels.endswith(".tsv") else ","
    )
    labels["Serotype"] = labels[label_column].fillna(missing_label)

    indices = (
        labels["Serotype"] != missing_label
        if args.labeled_only
        else np.ones(len(labels), dtype=bool)
    )
    if args.skip_labels:
        skip_indices = labels["Serotype"].isin(args.skip_labels)
        logger.info(
            "Skipping labels: %s accounting for %d samples.",
            args.skip_labels,
            skip_indices.sum(),
        )
        indices &= ~skip_indices

    fine_labels = labels["Serotype"][indices].values.tolist()
    coarse_labels = labels["Serotype"][indices].apply(map_serotype_to_group).tolist()

    # ── Classify each label as resolved serotype vs serogroup-only/compound ──
    label_types = [classify_label_type(lbl) for lbl in fine_labels]
    serotype_known = [lt == "serotype" for lt in label_types]

    # Count how many serogroup-only and compound labels we found
    n_serogroup = sum(1 for lt in label_types if lt == "serogroup_only")
    n_compound = sum(1 for lt in label_types if lt == "compound")
    logger.info(
        "Label classification: %d resolved serotypes, %d serogroup-only, %d compound",
        sum(serotype_known),
        n_serogroup,
        n_compound,
    )

    # ── Enforce minimum sample count per serotype ──
    min_count = args.model_params.get("min_serotype_count", MIN_SEROTYPE_COUNT)
    resolved_counts = Counter(lbl for lbl, sk in zip(fine_labels, serotype_known) if sk)
    rare_serotypes = {s for s, c in resolved_counts.items() if c < min_count}
    if rare_serotypes:
        logger.info(
            "Serotypes below min_count=%d (demoted to serogroup-only): %s with counts %s",
            min_count,
            sorted(rare_serotypes),
            [(s, resolved_counts[s]) for s in sorted(rare_serotypes)],
        )
        serotype_known = [
            (sk and fine_labels[i] not in rare_serotypes)
            for i, sk in enumerate(serotype_known)
        ]

    # ── Build serotype_to_idx from resolved labels ONLY ──
    resolved_serotypes = sorted(
        {lbl for lbl, sk in zip(fine_labels, serotype_known) if sk}
    )
    serotype_to_idx = {s: idx for idx, s in enumerate(resolved_serotypes)}
    num_serotypes = len(resolved_serotypes)
    logger.info(
        "Found %d resolved serotype classes: %s", num_serotypes, resolved_serotypes
    )

    # ── Genogroup index uses all capsulated samples (including serogroup-only) ──
    unique_genogroups = sorted(list(set(coarse_labels)))
    genogroup_to_idx = {group: idx for idx, group in enumerate(unique_genogroups)}
    num_genogroups = len(unique_genogroups)
    logger.info("Found %d unique genogroups: %s", num_genogroups, unique_genogroups)

    wandb.config.update(
        {
            "random_state": random_state,
            "k_folds": k_folds,
            "temperature": temperature,
            "weight_fine": weight_fine,
            "weight_coarse": weight_coarse,
            "num_layers": num_layers,
            "nhead": nhead,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "alpha": alpha,
            "output_dim": output_dim,
            "num_serotypes": num_serotypes,
            "num_genogroups": num_genogroups,
            "aug_noise_std": args.aug_noise_std,
            "aug_chunk_dropout": args.aug_chunk_dropout,
            "aug_spec_freq": args.aug_spec_freq,
            "aug_spec_width": args.aug_spec_width,
            "aug_n_views": args.aug_n_views,
        }
    )

    if args.hierarchical_loss:
        logger.info(
            "Using hierarchical contrastive loss with weights: %s, %s",
            weight_fine,
            weight_coarse,
        )
        labels_known = list(zip(coarse_labels, fine_labels))
        loss_function = partial(
            hierarchical_contrastive_loss,
            weight_fine=weight_fine,
            weight_coarse=weight_coarse,
        )
    else:
        labels_known = fine_labels
        loss_function = supervised_contrastive_loss

    sample_ids = (
        labels.index[indices] + CONTIG_SEP + labels["Contig_ID"][indices].astype(str)
    ).tolist()
    is_capsule = labels["Is_capsule"][indices].tolist()

    # Log overall class distributions for reproducibility
    sero_counts = Counter(fine_labels)
    geno_counts = Counter(coarse_labels)
    cbl_counts = Counter(is_capsule)
    wandb.log(
        {
            "class_dist/serotype_counts": wandb.Table(
                columns=["serotype", "count", "serotype_known"],
                data=[
                    [s, c, s in serotype_to_idx] for s, c in sorted(sero_counts.items())
                ],
            ),
            "class_dist/genogroup_counts": wandb.Table(
                columns=["genogroup", "count"],
                data=[[g, c] for g, c in sorted(geno_counts.items())],
            ),
            "class_dist/cbl_counts": wandb.Table(
                columns=["is_capsule", "count"],
                data=[[str(k), v] for k, v in sorted(cbl_counts.items())],
            ),
        }
    )

    # ── Stratification labels for k-fold ──
    # Use fine_label for resolved serotypes, coarse_label for serogroup-only/compound
    # to avoid StratifiedKFold failures on tiny classes.
    strat_labels = [
        fine_labels[i] if serotype_known[i] else f"__group__{coarse_labels[i]}"
        for i in range(len(fine_labels))
    ]

    serotype_known_arr = np.array(serotype_known)

    dataset_class = DatasetRegistry.get_dataset_class(dataset_name)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    cv_fold_metrics: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(
        skf.split(np.zeros(len(strat_labels)), strat_labels)
    ):
        logger.info("Fold %d / %d", fold + 1, k_folds)

        train_ds_base = dataset_class(
            embeddings_dir=args.embedding_dir,
            sample_ids=np.array(sample_ids)[train_idx],
            serotype_labels=np.array(labels_known)[train_idx],
            capsule_labels=np.array(is_capsule)[train_idx],
            serotype_known=serotype_known_arr[train_idx],
        )
        # Wrap training set with embedding augmentation if enabled
        if embedding_augmentor is not None:
            train_ds = AugmentedChunkedDataset(
                train_ds_base,
                augmentor=embedding_augmentor,
                n_views=args.aug_n_views,
            )
        else:
            train_ds = train_ds_base

        test_ds = dataset_class(
            embeddings_dir=args.embedding_dir,
            sample_ids=np.array(sample_ids)[test_idx],
            serotype_labels=np.array(labels_known)[test_idx],
            capsule_labels=np.array(is_capsule)[test_idx],
            serotype_known=serotype_known_arr[test_idx],
        )

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, collate_fn=collate_fn
        )

        # -- class weights from training split only (serotype weights from resolved-only samples) --
        train_fine = np.array(fine_labels)[train_idx]
        train_coarse = np.array(coarse_labels)[train_idx]
        train_capsule = np.array(is_capsule)[train_idx]
        train_sk = serotype_known_arr[train_idx]

        cbl_to_idx = {0: 0, 1: 1}
        cbl_weights = compute_class_weights(train_capsule.tolist(), cbl_to_idx, device)
        # Serotype weights computed from resolved-serotype samples only
        sero_weights = compute_class_weights(
            [lbl for lbl, sk in zip(train_fine, train_sk) if sk],
            serotype_to_idx,
            device,
        )
        geno_weights = compute_class_weights(
            train_coarse.tolist(), genogroup_to_idx, device
        )

        logger.info(
            "  Fold %d CBL weights:       %s", fold + 1, cbl_weights.cpu().tolist()
        )
        logger.info(
            "  Fold %d serotype weight range: [%.3f, %.3f]",
            fold + 1,
            sero_weights.min(),
            sero_weights.max(),
        )
        logger.info(
            "  Fold %d genogroup weight range: [%.3f, %.3f]",
            fold + 1,
            geno_weights.min(),
            geno_weights.max(),
        )

        model = TransformerTriHeadLR(
            input_dim=embedding_dim,
            num_classes=(num_serotypes, num_genogroups),
            output_dim=output_dim,
            nhead=nhead,
            num_layers=num_layers,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        ce_loss_fn = nn.CrossEntropyLoss(weight=cbl_weights)
        serotype_loss_fn = nn.CrossEntropyLoss(weight=sero_weights)
        genogroup_loss_fn = nn.CrossEntropyLoss(weight=geno_weights)

        best_loss, patience_counter = float("inf"), 0
        best_fold = {
            "fold": fold + 1,
            "epoch": 0,
            "val_loss": float("inf"),
            "cbl_accuracy": 0.0,
            "serotype_accuracy": 0.0,
            "genogroup_accuracy": 0.0,
            "n_train": int(len(train_idx)),
            "n_val": int(len(test_idx)),
        }

        for epoch in tqdm(range(args.epochs), desc=f"Training Fold {fold + 1}"):
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                ce_loss_fn,
                serotype_loss_fn,
                genogroup_loss_fn,
                loss_function,
                alpha,
                temperature,
                serotype_to_idx,
                genogroup_to_idx,
            )

            test_loss, cbl_accuracy, serotype_accuracy, genogroup_accuracy = evaluate(
                model,
                test_loader,
                ce_loss_fn,
                serotype_loss_fn,
                genogroup_loss_fn,
                loss_function,
                alpha,
                temperature,
                serotype_to_idx,
                genogroup_to_idx,
            )
            logger.info(
                "Fold %d - Epoch %d - Test Loss: %.4f, CBL Accuracy: %.4f, Serotype Accuracy: %.4f, Genogroup Accuracy: %.4f",
                fold + 1,
                epoch + 1,
                test_loss,
                cbl_accuracy,
                serotype_accuracy,
                genogroup_accuracy,
            )

            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
                best_fold = {
                    "fold": fold + 1,
                    "epoch": epoch + 1,
                    "val_loss": float(test_loss),
                    "cbl_accuracy": float(cbl_accuracy),
                    "serotype_accuracy": float(serotype_accuracy),
                    "genogroup_accuracy": float(genogroup_accuracy),
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(test_idx)),
                }
            else:
                patience_counter += 1
                if patience_counter >= args.early_stopping:
                    logger.info("Early stopping triggered.")
                    break
        cv_fold_metrics.append(best_fold)
        logger.info(
            "Fold %d best held-out — epoch %d: val_loss=%.4f, sero_acc=%.4f, geno_acc=%.4f, cbl_acc=%.4f",
            best_fold["fold"],
            best_fold["epoch"],
            best_fold["val_loss"],
            best_fold["serotype_accuracy"],
            best_fold["genogroup_accuracy"],
            best_fold["cbl_accuracy"],
        )

    # ── Persist held-out CV summary alongside the model ──
    cv_summary_path = os.path.splitext(args.output)[0] + "_cv_summary.json"
    sero_accs = [m["serotype_accuracy"] for m in cv_fold_metrics]
    geno_accs = [m["genogroup_accuracy"] for m in cv_fold_metrics]
    cbl_accs = [m["cbl_accuracy"] for m in cv_fold_metrics]
    cv_summary = {
        "k_folds": k_folds,
        "per_fold": cv_fold_metrics,
        "mean_serotype_accuracy": float(np.mean(sero_accs)) if sero_accs else 0.0,
        "std_serotype_accuracy": float(np.std(sero_accs)) if sero_accs else 0.0,
        "mean_genogroup_accuracy": float(np.mean(geno_accs)) if geno_accs else 0.0,
        "mean_cbl_accuracy": float(np.mean(cbl_accs)) if cbl_accs else 0.0,
    }
    with open(cv_summary_path, "w") as f:
        json.dump(cv_summary, f, indent=2)
    logger.info(
        "CV held-out: serotype %.4f ± %.4f over %d folds. Wrote summary to %s",
        cv_summary["mean_serotype_accuracy"],
        cv_summary["std_serotype_accuracy"],
        k_folds,
        cv_summary_path,
    )

    # Retrain on all data
    logger.info("Retraining on all data...")
    all_ds_base = dataset_class(
        embeddings_dir=args.embedding_dir,
        sample_ids=sample_ids,
        serotype_labels=labels_known,
        capsule_labels=is_capsule,
        serotype_known=serotype_known_arr,
    )
    if embedding_augmentor is not None:
        all_ds = AugmentedChunkedDataset(
            all_ds_base,
            augmentor=embedding_augmentor,
            n_views=args.aug_n_views,
        )
    else:
        all_ds = all_ds_base
    all_loader = DataLoader(
        all_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )

    # -- class weights from full dataset (serotype from resolved only) --
    cbl_to_idx = {0: 0, 1: 1}
    cbl_weights_all = compute_class_weights(is_capsule, cbl_to_idx, device)
    sero_weights_all = compute_class_weights(
        [lbl for lbl, sk in zip(fine_labels, serotype_known) if sk],
        serotype_to_idx,
        device,
    )
    geno_weights_all = compute_class_weights(coarse_labels, genogroup_to_idx, device)

    logger.info("  Final CBL weights:       %s", cbl_weights_all.cpu().tolist())
    logger.info(
        "  Final serotype weight range: [%.3f, %.3f]",
        sero_weights_all.min(),
        sero_weights_all.max(),
    )
    logger.info(
        "  Final genogroup weight range: [%.3f, %.3f]",
        geno_weights_all.min(),
        geno_weights_all.max(),
    )

    model_final = TransformerTriHeadLR(
        input_dim=embedding_dim,
        num_classes=(num_serotypes, num_genogroups),
        output_dim=output_dim,
        nhead=nhead,
        num_layers=num_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(model_final.parameters(), lr=args.lr)
    ce_loss_fn = nn.CrossEntropyLoss(weight=cbl_weights_all)
    serotype_loss_fn = nn.CrossEntropyLoss(weight=sero_weights_all)
    genogroup_loss_fn = nn.CrossEntropyLoss(weight=geno_weights_all)
    for epoch in tqdm(range(args.epochs), desc="Final Training"):
        train_one_epoch(
            model_final,
            all_loader,
            optimizer,
            ce_loss_fn,
            serotype_loss_fn,
            genogroup_loss_fn,
            loss_function,
            alpha,
            temperature,
            serotype_to_idx,
            genogroup_to_idx,
        )

    # Evaluate on clean (unaugmented) data for honest metrics
    logger.info("Evaluating final model on all data...")
    clean_loader = DataLoader(
        all_ds_base, batch_size=args.batch_size, collate_fn=collate_fn
    )
    (
        final_loss,
        final_cbl_accuracy,
        final_serotype_accuracy,
        final_genogroup_accuracy,
    ) = evaluate(
        model_final,
        clean_loader,
        ce_loss_fn,
        serotype_loss_fn,
        genogroup_loss_fn,
        loss_function,
        alpha,
        temperature,
        serotype_to_idx,
        genogroup_to_idx,
    )
    logger.info(
        "Final model - Loss: %.4f, CBL Accuracy: %.4f, Serotype Accuracy: %.4f, Genogroup Accuracy: %.4f",
        final_loss,
        final_cbl_accuracy,
        final_serotype_accuracy,
        final_genogroup_accuracy,
    )

    # Save model and serotype mapping
    model_save_dict = {
        "model_state_dict": model_final.state_dict(),
        "serotype_to_idx": serotype_to_idx,
        "genogroup_to_idx": genogroup_to_idx,
        "num_serotypes": num_serotypes,
        "num_genogroups": num_genogroups,
        "model_config": {
            "input_dim": embedding_dim,
            "num_classes": (num_serotypes, num_genogroups),
            "output_dim": output_dim,
            "nhead": nhead,
            "num_layers": num_layers,
        },
    }
    torch.save(model_save_dict, args.output)
    logger.info("Saved final model and serotype mapping to %s", args.output)


if __name__ == "__main__":
    args = parse_args()
    run_id = os.environ.get("SLURM_JOB_ID", os.urandom(4).hex())
    mode = "offline" if os.environ.get("WANDB_MODE") == "offline" else "online"
    wandb.init(project=WANDB_PROJECT_NAME, config=args, mode=mode)
    wandb.run.name = f"{WANDB_PROJECT_NAME}-{run_id}"
    main(args)
