import os
import glob
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .logging_config import get_logger

logger = get_logger(__name__)


def _make_padding_mask(x: torch.Tensor) -> torch.Tensor:
    """Detect all-zero (padding) positions BEFORE positional embedding.

    Args:
        x: (B, L, D) raw chunk embeddings from the base model.
    Returns:
        (B, L) boolean mask — ``True`` for padded (all-zero) positions.
    """
    return x.abs().sum(dim=-1) == 0  # (B, L)


def _masked_mean_pool(x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool over non-padding positions only.

    This fixes a consistency issue where ``collate_fn`` zero-pads shorter
    chunk sequences, but ``x.mean(dim=1)`` was averaging over all
    positions (including padding zeros), diluting the representation.
    Single-sample inference (query pipeline) had no padding and thus
    produced different pooled values for the same sample.

    Args:
        x: (B, L, D) encoded sequence (post-encoder).
        padding_mask: (B, L) boolean — ``True`` for padded positions.
    Returns:
        (B, D) mean-pooled tensor.
    """
    real = (~padding_mask).float().unsqueeze(-1)  # (B, L, 1)
    lengths = real.sum(dim=1).clamp(min=1)  # (B, 1)
    return (x * real).sum(dim=1) / lengths  # (B, D)


# Model Registry
class ModelRegistry:
    """Registry for model classes with factory pattern."""

    _models: Dict[str, Type["BaseModel"]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class."""

        def decorator(model_class: Type["BaseModel"]):
            cls._models[name] = model_class
            return model_class

        return decorator

    @classmethod
    def create_model(cls, name: str, **kwargs) -> "BaseModel":
        """Factory method to create a model instance."""
        if name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise ValueError(
                f"Model '{name}' not registered. Available models: {available}"
            )
        return cls._models[name](**kwargs)

    @classmethod
    def get_model_class(cls, name: str) -> Type["BaseModel"]:
        """Get the model class by name."""
        if name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise ValueError(
                f"Model '{name}' not registered. Available models: {available}"
            )
        return cls._models[name]


class DatasetRegistry:
    """Registry for dataset classes with factory pattern."""

    _datasets: Dict[str, Type[Dataset]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a dataset class."""

        def decorator(dataset_class: Type[Dataset]):
            cls._datasets[name] = dataset_class
            return dataset_class

        return decorator

    @classmethod
    def create_dataset(cls, name: str, **kwargs) -> Dataset:
        """Factory method to create a dataset instance."""
        if name not in cls._datasets:
            available = ", ".join(cls._datasets.keys())
            raise ValueError(
                f"Dataset '{name}' not registered. Available datasets: {available}"
            )
        return cls._datasets[name](**kwargs)

    @classmethod
    def get_dataset_class(cls, name: str) -> Type[Dataset]:
        """Get the dataset class by name."""
        if name not in cls._datasets:
            available = ", ".join(cls._datasets.keys())
            raise ValueError(
                f"Dataset '{name}' not registered. Available datasets: {available}"
            )
        return cls._datasets[name]


# Convenience decorator
def register_model(name: str):
    """Convenience decorator for registering models."""
    return ModelRegistry.register(name)


# TODO merge with model registry
def register_dataset(name: str):
    """Convenience decorator for registering datasets."""
    return DatasetRegistry.register(name)


# Abstract Base Model
class BaseModel(nn.Module, ABC):
    """Abstract base class for all models."""

    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        """Forward pass of the model."""
        pass

    def get_model_info(self) -> dict:
        """Get model information including parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_class": self.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }

    @classmethod
    def from_config(cls, config: dict):
        """Create model instance from configuration dictionary."""
        return cls(**config)


@register_model("contrastive_head")
class ContrastiveHead(BaseModel):
    def __init__(self, input_dim, output_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)


@register_dataset("contrastive_chunked")
class ContrastiveChunkedDataset(Dataset):
    def __init__(
        self,
        embeddings_dir,
        sample_ids,
        serotype_labels,
        capsule_labels,
        serotype_known=None,
    ):
        """
        embeddings_path: str, path to directory with npy entries of variable length chunked embeddings
        serotype_labels: pd.DataFrame, DataFrame with serotype labels indexed by sample ID.
        serotype_known: optional array-like of bool, True if the serotype is resolved (not serogroup/compound).
        """
        self.embedding_dir = embeddings_dir
        self.serotypes = serotype_labels
        self.is_capsule = capsule_labels
        self.sample_ids = sample_ids
        self.serotype_known = (
            serotype_known if serotype_known is not None else [True] * len(sample_ids)
        )

        # TODO Validate sub-folders too
        all_embeddings = glob.glob(os.path.join(embeddings_dir, "**/*.npy"))
        file_names = [os.path.basename(f).split(".")[0] for f in all_embeddings]

        missing_samples = set(self.sample_ids) - set(file_names)
        if missing_samples:
            logger.warning(
                "%d sample_ids do not have a corresponding embedding file: %s",
                len(missing_samples),
                missing_samples,
            )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx):
        subdir = "cbl" if self.is_capsule[idx] else "non-cbl"
        embedding_path = os.path.join(
            self.embedding_dir, subdir, f"{self.sample_ids[idx]}.npy"
        )
        return {
            "sample_id": self.sample_ids[idx],
            "embedding": torch.tensor(np.load(embedding_path), dtype=torch.float32),
            "serotype": self.serotypes[idx],
            "is_capsule": self.is_capsule[idx],
            "serotype_known": self.serotype_known[idx],
        }


@register_dataset("multidomain_chunked")
class MultidomainChunkedDataset(Dataset):
    def __init__(
        self,
        embeddings_dir,
        sample_ids,
        serotype_labels,
        capsule_labels,
        serotype_known=None,
    ):
        """
        embeddings_path: str, path to directory with npy entries of variable length chunked embeddings
        serotype_labels: pd.DataFrame, DataFrame with serotype labels indexed by sample ID.
        serotype_known: optional array-like of bool, True if the serotype is resolved (not serogroup/compound).
        """
        self.embedding_dir = embeddings_dir
        self.serotypes = serotype_labels
        self.is_capsule = capsule_labels
        self.sample_ids = sample_ids
        self.serotype_known = (
            serotype_known if serotype_known is not None else [True] * len(sample_ids)
        )

        # TODO Validate sub-folders too
        all_embeddings = glob.glob(os.path.join(embeddings_dir, "*.npy"))
        file_names = [os.path.basename(f).split(".")[0] for f in all_embeddings]

        missing_samples = set(self.sample_ids) - set(file_names)
        if missing_samples:
            logger.warning(
                "%d sample_ids do not have a corresponding embedding file: %s",
                len(missing_samples),
                missing_samples,
            )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx):
        # subdir = "cbl" if self.is_capsule[idx] else "non-cbl"
        embedding_path = os.path.join(self.embedding_dir, f"{self.sample_ids[idx]}.npy")
        return {
            "sample_id": self.sample_ids[idx],
            "embedding": torch.tensor(np.load(embedding_path), dtype=torch.float32),
            "serotype": self.serotypes[idx],
            "is_capsule": self.is_capsule[idx],
            "serotype_known": self.serotype_known[idx],
        }


@register_model("transformer_contrastive_head")
class TransformerContrastiveHead(BaseModel):
    def __init__(
        self, input_dim, output_dim=128, max_len=64, nhead=4, num_layers=2, **kwargs
    ):
        super().__init__(**kwargs)
        self.pos_embed = nn.Embedding(
            max_len, input_dim
        )  # TODO Dynamically expand or clamp

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=4 * input_dim,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cbl_classifier = nn.Linear(output_dim, 2)

        self.project = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        B, L, D = x.size()
        padding_mask = _make_padding_mask(x)  # (B, L)
        pos = torch.arange(L, device=x.device).unsqueeze(0)  # (1, L)
        x = x + self.pos_embed(pos)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = _masked_mean_pool(x, padding_mask)
        z = F.normalize(self.project(x), dim=1)
        logits = self.cbl_classifier(z)  # Classifier output (B, output_dim)
        return logits, z


@register_model("transformer_lr_classifier")
class TransformerLRClassifier(BaseModel):
    def __init__(
        self,
        input_dim,
        num_classes,
        output_dim=128,
        max_len=64,
        nhead=4,
        num_layers=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pos_embed = nn.Embedding(max_len, input_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=4 * input_dim,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cbl_classifier = nn.Linear(output_dim, 2)
        self.serotype_classifier = nn.Linear(output_dim, num_classes)

        self.project = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        B, L, D = x.size()
        padding_mask = _make_padding_mask(x)  # (B, L)
        pos = torch.arange(L, device=x.device).unsqueeze(0)  # (1, L)
        x = x + self.pos_embed(pos)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = _masked_mean_pool(x, padding_mask)
        z = F.normalize(self.project(x), dim=1)
        logits = self.cbl_classifier(z)  # Classifier output (B, output_dim)
        serotype_logits = self.serotype_classifier(z)
        return logits, serotype_logits, z


@register_model("transformer_trihead_lr")
class TransformerTriHeadLR(BaseModel):
    def __init__(
        self,
        input_dim,
        num_classes,
        output_dim=128,
        max_len=64,
        nhead=4,
        num_layers=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pos_embed = nn.Embedding(max_len, input_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=4 * input_dim,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cbl_classifier = nn.Linear(output_dim, 2)

        assert len(num_classes) == 2, (
            "num_classes should be a list or tuple of length 2 for the two classifiers."
        )
        self.serotype_classifier = nn.Linear(output_dim, num_classes[0])
        self.genogroup_classifier = nn.Linear(output_dim, num_classes[1])

        self.project = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        B, L, D = x.size()
        padding_mask = _make_padding_mask(x)  # (B, L)
        pos = torch.arange(L, device=x.device).unsqueeze(0)  # (1, L)
        x = x + self.pos_embed(pos)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = _masked_mean_pool(x, padding_mask)
        z = F.normalize(self.project(x), dim=1)
        logits = self.cbl_classifier(z)  # Classifier output (B, output_dim)
        serotype_logits = self.serotype_classifier(z)
        genogroup_logits = self.genogroup_classifier(z)
        return logits, (serotype_logits, genogroup_logits), z
