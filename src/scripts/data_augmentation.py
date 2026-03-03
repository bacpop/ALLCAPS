"""DNA sequence augmentation for robust transformer-based serotyping.

Augmentations target assembler-induced variability (e.g. Velvet vs SPAdes)
and general sequencing noise so that the embedding model becomes invariant
to these artefacts rather than memorising assembler-specific patterns.

Two levels of augmentation are provided:

1. **Sequence-level** (``SequenceAugmentor``): applied to raw DNA *before*
   the base-model tokenizer.  Use offline via the CLI to produce augmented
   FASTA files, or on-the-fly inside a custom Dataset.

2. **Embedding-level** (``EmbeddingAugmentor``): applied to the pre-computed
   per-chunk ``.npy`` embeddings *during* training, avoiding the cost of
   re-running the base model.  Includes Gaussian noise, chunk-dropout, and
   mix-up.

A drop-in ``AugmentedChunkedDataset`` wraps an existing
``ContrastiveChunkedDataset`` and applies embedding-level augmentation
on-the-fly.

CLI usage (offline FASTA augmentation)::

    python -m scripts.data_augmentation \\
        --fasta data/cps_cleaned.fasta \\
        --output data/cps_augmented.fasta \\
        --n_augments 3 \\
        --sub_rate 0.005 --indel_rate 0.001 --rc_prob 0.5

Best practices
--------------
* Keep substitution rate low (0.1-0.5 %) — typical short-read error rates.
* Indel rate should be even lower (~0.05-0.1 %) for assembler-level noise.
* Reverse-complement augmentation is free and strongly recommended for
  bacterial genomics (strand is arbitrary after assembly).
* Combine with embedding-level Gaussian noise (σ ≈ 0.01-0.05) during
  training for additional regularisation.
* For the CBL / non-CBL split, apply the *same* augmentor to both classes
  to preserve the distributional relationship.
"""

from __future__ import annotations

import argparse
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

_BASES = "ACGT"
_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


# ═══════════════════════════════════════════════════════════════
#  Low-level DNA transforms
# ═══════════════════════════════════════════════════════════════


def reverse_complement(seq: str) -> str:
    """Return reverse-complement of a DNA string."""
    return seq.translate(_COMPLEMENT)[::-1]


def random_substitutions(seq: str, rate: float, rng: random.Random) -> str:
    """Replace each base independently with probability *rate*.

    The replacement base is chosen uniformly from the three alternatives.
    """
    if rate <= 0:
        return seq
    out: list = list(seq)
    for i in range(len(out)):
        if rng.random() < rate:
            alts = [b for b in _BASES if b != out[i].upper()]
            out[i] = rng.choice(alts)
    return "".join(out)


def random_indels(seq: str, rate: float, rng: random.Random) -> str:
    """Apply random insertions and deletions at *rate* per position.

    Each position has ``rate/2`` chance of insertion (random base before it)
    and ``rate/2`` chance of deletion.
    """
    if rate <= 0:
        return seq
    half = rate / 2.0
    out: list = []
    for ch in seq:
        if rng.random() < half:
            # Insertion: add a random base before this position
            out.append(rng.choice(_BASES))
        if rng.random() >= half:
            out.append(ch)
        # else: deletion — skip this character
    return "".join(out)


def random_crop(
    seq: str,
    frac_range: Tuple[float, float],
    rng: random.Random,
) -> str:
    """Crop a random fraction from a random end (start or end).

    Simulates different contig breakpoints across assemblers.
    """
    lo, hi = frac_range
    frac = rng.uniform(lo, hi)
    n_crop = max(1, int(len(seq) * frac))
    if rng.random() < 0.5:
        return seq[n_crop:]  # trim from start
    return seq[:-n_crop]  # trim from end


def random_mask(seq: str, frac: float, kmer: int, rng: random.Random) -> str:
    """Mask random k-mers with N's.

    Simulates low-confidence / ambiguous regions that differ between
    assemblers.
    """
    if frac <= 0:
        return seq
    n_mask = max(1, int(len(seq) * frac / kmer))
    out = list(seq)
    for _ in range(n_mask):
        pos = rng.randint(0, max(0, len(out) - kmer))
        for j in range(pos, min(pos + kmer, len(out))):
            out[j] = "N"
    return "".join(out)


def random_contig_break(
    seq: str,
    n_breaks: int,
    jitter: int,
    rng: random.Random,
) -> str:
    """Simulate contig breaks by inserting random bases at break-points.

    Each break inserts *jitter* random bases, mimicking different assembly
    graph resolutions across tools.
    """
    if n_breaks <= 0 or jitter <= 0:
        return seq
    positions = sorted(rng.sample(range(1, len(seq)), min(n_breaks, len(seq) - 1)))
    parts: list = []
    prev = 0
    for p in positions:
        parts.append(seq[prev:p])
        parts.append("".join(rng.choice(_BASES) for _ in range(jitter)))
        prev = p
    parts.append(seq[prev:])
    return "".join(parts)


# ═══════════════════════════════════════════════════════════════
#  SequenceAugmentor  (DNA-level, pre-tokeniser)
# ═══════════════════════════════════════════════════════════════


@dataclass
class SequenceAugmentor:
    """Configurable DNA augmentation pipeline.

    All transforms are stochastic and applied independently each time
    ``__call__`` is invoked, so calling it multiple times on the same
    sequence will produce different augmented versions.

    Parameters
    ----------
    sub_rate : float
        Per-base substitution probability (default 0.5 %).
    indel_rate : float
        Per-base insertion/deletion probability (default 0.1 %).
    rc_prob : float
        Probability of applying reverse-complement (default 0 %).
    crop_prob : float
        Probability of random boundary cropping (default 30 %).
    crop_frac : tuple of (lo, hi)
        Fraction of sequence to crop when cropping is applied.
    mask_prob : float
        Probability of masking random k-mers with N's.
    mask_frac : float
        Fraction of sequence to mask.
    mask_kmer : int
        k-mer size for masking.
    contig_break_prob : float
        Probability of inserting simulated contig breaks.
    contig_break_n : int
        Number of break-points to insert.
    contig_break_jitter : int
        Number of random bases inserted at each break-point.
    seed : int or None
        Random seed for reproducibility (None = non-deterministic).
    """

    sub_rate: float = 0.005
    indel_rate: float = 0.001
    rc_prob: float = 0.0
    crop_prob: float = 0.3
    crop_frac: Tuple[float, float] = (0.05, 0.15)
    mask_prob: float = 0.0
    mask_frac: float = 0.01
    mask_kmer: int = 6
    contig_break_prob: float = 0.0
    contig_break_n: int = 2
    contig_break_jitter: int = 10
    seed: Optional[int] = None

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def __call__(self, seq: str) -> str:
        """Apply a random combination of augmentations."""
        # Reverse complement
        if self.rc_prob > 0 and self._rng.random() < self.rc_prob:
            seq = reverse_complement(seq)

        # Contig boundary cropping
        if self.crop_prob > 0 and self._rng.random() < self.crop_prob:
            seq = random_crop(seq, self.crop_frac, self._rng)

        # Nucleotide substitution
        seq = random_substitutions(seq, self.sub_rate, self._rng)

        # Insertions / deletions
        seq = random_indels(seq, self.indel_rate, self._rng)

        # k-mer masking
        if self.mask_prob > 0 and self._rng.random() < self.mask_prob:
            seq = random_mask(seq, self.mask_frac, self.mask_kmer, self._rng)

        # Simulated contig breaks
        if self.contig_break_prob > 0 and self._rng.random() < self.contig_break_prob:
            seq = random_contig_break(
                seq, self.contig_break_n, self.contig_break_jitter, self._rng
            )

        return seq


# ═══════════════════════════════════════════════════════════════
#  EmbeddingAugmentor  (embedding-level, post-base-model)
# ═══════════════════════════════════════════════════════════════


@dataclass
class EmbeddingAugmentor:
    """Augmentation applied to pre-computed chunk embeddings during training.

    These transforms avoid the cost of re-running the base language model
    and instead add noise directly in embedding space.

    Parameters
    ----------
    noise_std : float
        Standard deviation of additive Gaussian noise (0 = disabled).
    chunk_dropout : float
        Probability of dropping each chunk entirely (replaced by zeros).
        Must be < 1.0 to keep at least one chunk.
    spec_augment_freq : float
        Probability of applying SpecAugment-style feature masking
        (mask contiguous feature dimensions with zeros).
    spec_augment_width : int
        Maximum width (in feature dims) of each SpecAugment mask.
    """

    noise_std: float = 0.01
    chunk_dropout: float = 0.1
    spec_augment_freq: float = 0.0
    spec_augment_width: int = 16

    def __call__(
        self, embedding: torch.Tensor, training: bool = True
    ) -> torch.Tensor:
        """Augment a (L, D) embedding tensor in-place.

        Returns the augmented tensor (same shape).
        Only active when ``training=True``.
        """
        if not training:
            return embedding

        L, D = embedding.shape

        # Gaussian noise
        if self.noise_std > 0:
            embedding = embedding + torch.randn_like(embedding) * self.noise_std

        # Chunk dropout (keep at least 1)
        if self.chunk_dropout > 0 and L > 1:
            mask = torch.rand(L) > self.chunk_dropout
            if not mask.any():
                mask[torch.randint(L, (1,))] = True
            embedding = embedding * mask.unsqueeze(-1).float()

        # SpecAugment-style feature masking
        if self.spec_augment_freq > 0 and random.random() < self.spec_augment_freq:
            width = random.randint(1, self.spec_augment_width)
            start = random.randint(0, max(0, D - width))
            embedding[:, start : start + width] = 0

        return embedding


# ═══════════════════════════════════════════════════════════════
#  AugmentedChunkedDataset  (drop-in replacement for training)
# ═══════════════════════════════════════════════════════════════


class AugmentedChunkedDataset(Dataset):
    """Wraps an existing chunked dataset and applies embedding augmentation.

    Usage::

        from scripts.models import DatasetRegistry
        from scripts.data_augmentation import AugmentedChunkedDataset, EmbeddingAugmentor

        base_ds = DatasetRegistry.get_dataset_class("contrastive_chunked")(...)
        aug = EmbeddingAugmentor(noise_std=0.02, chunk_dropout=0.15)
        train_ds = AugmentedChunkedDataset(base_ds, augmentor=aug, n_views=2)

    When ``n_views > 1``, each ``__getitem__`` returns the original plus
    ``n_views - 1`` augmented copies (useful for contrastive learning with
    augmented positives).
    """

    def __init__(
        self,
        base_dataset: Dataset,
        augmentor: EmbeddingAugmentor,
        n_views: int = 1,
    ):
        self.base = base_dataset
        self.aug = augmentor
        self.n_views = max(1, n_views)

    def __len__(self) -> int:
        return len(self.base) * self.n_views  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        real_idx = idx % len(self.base)  # type: ignore[arg-type]
        item = deepcopy(self.base[real_idx])

        # First view is always unaugmented
        if idx // len(self.base) > 0:  # type: ignore[arg-type]
            item["embedding"] = self.aug(item["embedding"], training=True)

        return item


# ═══════════════════════════════════════════════════════════════
#  Mixup utility
# ═══════════════════════════════════════════════════════════════


def mixup_embeddings(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, float]:
    """Create a mix-up interpolation between two embeddings.

    Returns the interpolated embedding and the lambda coefficient.
    Useful for creating virtual training examples between similar serotypes.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    mixed = lam * emb_a + (1 - lam) * emb_b
    return mixed, lam


# ═══════════════════════════════════════════════════════════════
#  CLI — offline FASTA augmentation
# ═══════════════════════════════════════════════════════════════


def main():
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    parser = argparse.ArgumentParser(
        description="Augment a FASTA file with DNA-level transforms."
    )
    parser.add_argument("--fasta", required=True, help="Input FASTA file")
    parser.add_argument("--output", required=True, help="Output augmented FASTA")
    parser.add_argument(
        "--n_augments",
        type=int,
        default=3,
        help="Number of augmented copies per sequence (default: 3)",
    )
    parser.add_argument("--sub_rate", type=float, default=0.005)
    parser.add_argument("--indel_rate", type=float, default=0.001)
    parser.add_argument("--rc_prob", type=float, default=0.5)
    parser.add_argument("--crop_prob", type=float, default=0.3)
    parser.add_argument(
        "--crop_frac",
        type=str,
        default="0.05,0.15",
        help="lo,hi fraction for random crop (default: 0.05,0.15)",
    )
    parser.add_argument("--mask_prob", type=float, default=0.0)
    parser.add_argument("--mask_frac", type=float, default=0.01)
    parser.add_argument("--mask_kmer", type=int, default=6)
    parser.add_argument("--contig_break_prob", type=float, default=0.0)
    parser.add_argument("--contig_break_n", type=int, default=2)
    parser.add_argument("--contig_break_jitter", type=int, default=10)
    parser.add_argument(
        "--include_original",
        action="store_true",
        help="Include the original (unaugmented) sequences in the output.",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    crop_lo, crop_hi = map(float, args.crop_frac.split(","))

    augmentor = SequenceAugmentor(
        sub_rate=args.sub_rate,
        indel_rate=args.indel_rate,
        rc_prob=args.rc_prob,
        crop_prob=args.crop_prob,
        crop_frac=(crop_lo, crop_hi),
        mask_prob=args.mask_prob,
        mask_frac=args.mask_frac,
        mask_kmer=args.mask_kmer,
        contig_break_prob=args.contig_break_prob,
        contig_break_n=args.contig_break_n,
        contig_break_jitter=args.contig_break_jitter,
        seed=args.seed,
    )

    records_in = list(SeqIO.parse(args.fasta, "fasta"))
    records_out: List[SeqRecord] = []

    for rec in records_in:
        seq_str = str(rec.seq)

        if args.include_original:
            records_out.append(rec)

        for k in range(args.n_augments):
            aug_seq = augmentor(seq_str)
            aug_id = f"{rec.id}_aug{k}"
            aug_rec = SeqRecord(
                Seq(aug_seq),
                id=aug_id,
                description=f"augmented_from={rec.id} aug_idx={k}",
            )
            records_out.append(aug_rec)

    with open(args.output, "w") as fh:
        SeqIO.write(records_out, fh, "fasta")

    n_orig = len(records_in)
    n_out = len(records_out)
    print(
        f"Wrote {n_out} records ({n_orig} originals × {args.n_augments} augments"
        f"{' + originals' if args.include_original else ''}) → {args.output}"
    )


if __name__ == "__main__":
    main()
