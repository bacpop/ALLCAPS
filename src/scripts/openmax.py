#!/usr/bin/env python
"""
OpenMax — Open-Set Recognition for serotype classification.

Implements Bendale & Boult (2016) "Towards Open Set Deep Networks":
  1. fit():  Compute per-class Mean Activation Vectors (MAVs) and fit Weibull
             distributions to the tail of per-class distances (EVT).
  2. openmax_score():  Recalibrate logits by redistributing probability mass
             to an "UNKNOWN" class based on Weibull CDF of distance to MAV.
  3. CLI modes:  --fit (calibrate from .npz + labels + model)
                 --predict (score from .npz or query results)

The key insight is that correctly classified in-distribution samples cluster
tightly around their class MAV, while OOD / novel-class samples are further
away.  The Weibull tail models how extreme those distances get for known
classes, enabling a principled "unknown" probability.

Usage:
  # Fit (calibrate):
  python -m scripts.openmax fit \
      --embeddings results/inference_results.npz \
      --labels results/final_metadata.tsv \
      --model results/transformer_model.pth \
      --output results/openmax_params.pkl \
      --device cpu

  # Predict:
  python -m scripts.openmax predict \
      --embeddings results/inference_results.npz \
      --labels results/final_metadata.tsv \
      --model results/transformer_model.pth \
      --openmax_params results/openmax_params.pkl \
      --output results/openmax_predictions.csv \
      --device cpu
"""

import argparse
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import exponweib
from tqdm import tqdm

from .consts import (
    DEFAULT_BATCH_SIZE, DEFAULT_HEAD_MODEL, DEFAULT_SEP,
    DEFAULT_MISSING_LABEL
)
from .models import ModelRegistry
from .utils import get_sample_id


# ──────────────────────────── Core OpenMax ────────────────────────────

class OpenMax:
    """
    OpenMax open-set recogniser.

    Attributes
    ----------
    mavs : dict[int, np.ndarray]
        Mean Activation Vector per class index.
    weibull_params : dict[int, tuple]
        (shape_a, shape_c, loc, scale) of the fitted ``exponweib`` for each class.
    alpha : int
        Number of top-ranked classes to revise (α in the paper).
    tail_size : int
        Number of largest distances used for Weibull fitting (η).
    distance_metric : str
        'euclid' or 'cosine'.
    idx_to_class : dict[int, str]
        Reverse map from class index to class name.
    """

    UNKNOWN_LABEL = "UNKNOWN"

    def __init__(
        self,
        alpha: int = 10,
        tail_size: int = 20,
        distance_metric: str = "euclid",
    ):
        self.alpha = alpha
        self.tail_size = tail_size
        self.distance_metric = distance_metric

        self.mavs: Dict[int, np.ndarray] = {}
        self.weibull_params: Dict[int, Tuple[float, float, float, float]] = {}
        self.idx_to_class: Dict[int, str] = {}
        self._fitted = False

    # ─────────── distance helpers ───────────

    @staticmethod
    def _euclid_dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    @staticmethod
    def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-12:
            return 1.0
        return float(1.0 - np.dot(a, b) / denom)

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.distance_metric == "cosine":
            return self._cosine_dist(a, b)
        return self._euclid_dist(a, b)

    # ─────────── fitting ───────────

    def fit(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        class_to_idx: Dict[str, int],
    ) -> "OpenMax":
        """
        Fit Weibull models on training-set activation vectors.

        Parameters
        ----------
        activations : (N, D) array
            Activation vectors (e.g., L2-normalised ``z`` embeddings) for
            *correctly classified* training samples.
        labels : (N,) array of str
            Ground-truth class labels aligned with ``activations``.
        class_to_idx : dict
            Mapping from class name → integer index.

        Returns
        -------
        self : OpenMax (fitted)
        """
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        idx_arr = np.array([class_to_idx[lbl] for lbl in labels])
        unique_classes = sorted(set(idx_arr))

        print(f"Fitting OpenMax on {len(activations)} samples, {len(unique_classes)} classes "
              f"(tail_size={self.tail_size}, alpha={self.alpha}, distance={self.distance_metric})")

        for cls_idx in tqdm(unique_classes, desc="Fitting Weibull per class"):
            mask = idx_arr == cls_idx
            cls_acts = activations[mask]
            if len(cls_acts) < 2:
                # Too few samples; skip (will not produce Weibull)
                print(f"  Warning: class {self.idx_to_class.get(cls_idx, cls_idx)} has <2 samples, skipping Weibull fit.")
                continue

            mav = cls_acts.mean(axis=0)
            self.mavs[cls_idx] = mav

            # Compute distances of correctly-classified samples to their own class MAV
            dists = np.array([self._dist(act, mav) for act in cls_acts])

            # Fit Weibull to the tail (η largest distances)
            tail = np.sort(dists)[-self.tail_size:]
            if len(tail) < 3:
                tail = np.sort(dists)  # use all if very small class

            try:
                params = exponweib.fit(tail, floc=0)  # (a, c, loc, scale)
                self.weibull_params[cls_idx] = params
            except Exception as e:
                print(f"  Warning: Weibull fit failed for class {self.idx_to_class.get(cls_idx, cls_idx)}: {e}")
                # Fallback: use a very loose Weibull (never triggers unknown)
                self.weibull_params[cls_idx] = (1.0, 1.0, 0.0, 1e6)

        self._fitted = True
        print(f"Fitted Weibull for {len(self.weibull_params)} / {len(unique_classes)} classes.")
        return self

    # ─────────── scoring ───────────

    def _alpha_weights(self, n_classes: int) -> np.ndarray:
        """Linearly decaying weights for the top-α classes (Eq. 3 in paper).
        α_j = 1 - j/α for j = 0, 1, ..., α-1  (1-indexed becomes 0-indexed)."""
        alpha = min(self.alpha, n_classes)
        return np.array([1.0 - (j / alpha) for j in range(alpha)])

    def openmax_score(
        self,
        activation: np.ndarray,
        logits: np.ndarray,
    ) -> Dict:
        """
        Compute OpenMax recalibrated probabilities for a single sample.

        Parameters
        ----------
        activation : (D,) array
            The L2-normalised projected ``z`` embedding.
        logits : (K,) array
            Raw serotype classifier logits.

        Returns
        -------
        dict with keys:
            'openmax_probs'     : (K+1,) array — recalibrated probs incl. "UNKNOWN"
            'openmax_pred'      : str — predicted class name or "UNKNOWN"
            'prob_unknown'      : float — probability assigned to UNKNOWN
            'is_novel'          : bool — True if top-1 is UNKNOWN
            'raw_softmax_pred'  : str — the closed-set softmax prediction for reference
        """
        assert self._fitted, "OpenMax not fitted. Call .fit() first."

        K = len(logits)
        logits = logits.astype(np.float64)

        # Compute distances to all MAVs
        distances = {}
        for cls_idx, mav in self.mavs.items():
            distances[cls_idx] = self._dist(activation, mav)

        # Rank classes by logit magnitude (descending)
        ranked_indices = np.argsort(-logits)  # highest logit first
        alpha = min(self.alpha, K)
        weights = self._alpha_weights(K)

        # Revised logits (OpenMax "activation vector")
        revised_logits = logits.copy()
        unknown_logit = 0.0

        for rank, cls_idx in enumerate(ranked_indices[:alpha]):
            cls_idx = int(cls_idx)
            if cls_idx not in self.weibull_params:
                continue

            dist = distances.get(cls_idx, 1e6)
            a, c, loc, scale = self.weibull_params[cls_idx]
            # Weibull CDF: probability that the distance is <= observed
            w = float(exponweib.cdf(dist, a, c, loc=loc, scale=scale))

            # w close to 1 → sample is very far from this class MAV → likely OOD
            rev_weight = weights[rank] * w
            revised_logits[cls_idx] = logits[cls_idx] * (1.0 - rev_weight)
            unknown_logit += logits[cls_idx] * rev_weight

        # Append unknown logit and compute softmax
        extended = np.append(revised_logits, unknown_logit)
        # Numerically stable softmax
        extended_shifted = extended - extended.max()
        exp_ext = np.exp(extended_shifted)
        openmax_probs = exp_ext / exp_ext.sum()

        prob_unknown = float(openmax_probs[-1])
        pred_idx = int(np.argmax(openmax_probs))

        if pred_idx == K:
            pred_label = self.UNKNOWN_LABEL
            is_novel = True
        else:
            pred_label = self.idx_to_class.get(pred_idx, f"class_{pred_idx}")
            is_novel = False

        # Closed-set reference
        raw_probs = np.exp(logits - logits.max())
        raw_probs = raw_probs / raw_probs.sum()
        raw_pred_idx = int(np.argmax(raw_probs))
        raw_pred_label = self.idx_to_class.get(raw_pred_idx, f"class_{raw_pred_idx}")

        return {
            "openmax_probs": openmax_probs,
            "openmax_pred": pred_label,
            "prob_unknown": prob_unknown,
            "is_novel": is_novel,
            "raw_softmax_pred": raw_pred_label,
        }

    def batch_score(
        self,
        activations: np.ndarray,
        logits: np.ndarray,
    ) -> List[Dict]:
        """Score a batch of samples. Returns list of dicts from openmax_score()."""
        results = []
        for i in range(len(activations)):
            results.append(self.openmax_score(activations[i], logits[i]))
        return results

    # ─────────── persistence ───────────

    def save(self, path: str):
        """Save fitted parameters to pickle."""
        assert self._fitted, "Cannot save unfitted OpenMax."
        data = {
            "mavs": self.mavs,
            "weibull_params": self.weibull_params,
            "idx_to_class": self.idx_to_class,
            "alpha": self.alpha,
            "tail_size": self.tail_size,
            "distance_metric": self.distance_metric,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"OpenMax parameters saved to {path}")

    @classmethod
    def load(cls, path: str) -> "OpenMax":
        """Load fitted parameters from pickle."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(
            alpha=data["alpha"],
            tail_size=data["tail_size"],
            distance_metric=data["distance_metric"],
        )
        obj.mavs = data["mavs"]
        obj.weibull_params = data["weibull_params"]
        obj.idx_to_class = data["idx_to_class"]
        obj._fitted = True
        return obj


# ──────────────────── CLI: fit mode ────────────────────

def cli_fit(args):
    """Fit OpenMax from pre-computed .npz embeddings (z vectors) + model + labels."""
    device = torch.device(args.device)
    sep = args.sep

    # Load model
    model_save_dict = torch.load(args.model, map_location=device)
    model_config = model_save_dict["model_config"]
    serotype_to_idx = model_save_dict["serotype_to_idx"]
    head_model = args.head_model

    model = ModelRegistry.get_model_class(head_model).from_config(model_config).to(device)
    model.load_state_dict(model_save_dict["model_state_dict"])
    model.eval()

    # Load embeddings + labels
    X = np.load(args.embeddings, allow_pickle=True)
    labels_df = pd.read_csv(args.labels, index_col=0, sep="\t" if args.labels.endswith(".tsv") else ",")
    labels_df["Serotype"] = labels_df["Serotype"].fillna(DEFAULT_MISSING_LABEL)
    labels_df = labels_df[labels_df["Serotype"] != DEFAULT_MISSING_LABEL]
    # Only capsulated samples for serotype
    labels_df = labels_df[labels_df["Is_capsule"].astype(bool)]

    keys = (
        labels_df["Is_capsule"].map(lambda x: "cbl" if x else "non-cbl")
        + sep
        + get_sample_id(labels_df)
    )

    valid_keys = [k for k in keys if k in X]
    valid_mask = keys.isin(valid_keys)
    labels_df = labels_df[valid_mask.values]
    keys = keys[valid_mask.values]

    X_filtered = np.stack([X[k] for k in keys])
    serotype_labels = labels_df["Serotype"].values
    print(f"Loaded {len(X_filtered)} capsulated embeddings for fitting.")

    # Compute serotype logits and filter to correctly classified
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(X_filtered), args.batch_size):
            batch = torch.tensor(X_filtered[i : i + args.batch_size], dtype=torch.float32, device=device)
            sero_logits = model.serotype_classifier(batch)
            all_logits.append(sero_logits.cpu().numpy())
    all_logits = np.concatenate(all_logits, axis=0)

    # Filter to correctly classified
    pred_indices = np.argmax(all_logits, axis=1)
    true_indices = np.array([serotype_to_idx.get(lbl, -1) for lbl in serotype_labels])
    correct_mask = pred_indices == true_indices

    print(f"Correctly classified: {correct_mask.sum()}/{len(correct_mask)} "
          f"({correct_mask.mean():.2%})")

    activations_correct = X_filtered[correct_mask]
    labels_correct = serotype_labels[correct_mask]

    # Filter to classes present in the class_to_idx
    valid_label_mask = np.array([lbl in serotype_to_idx for lbl in labels_correct])
    activations_correct = activations_correct[valid_label_mask]
    labels_correct = labels_correct[valid_label_mask]

    # Fit OpenMax
    om = OpenMax(
        alpha=args.alpha,
        tail_size=args.tail_size,
        distance_metric=args.distance_metric,
    )
    om.fit(activations_correct, labels_correct, serotype_to_idx)
    om.save(args.output)


# ──────────────────── CLI: predict mode ────────────────────

def cli_predict(args):
    """Score samples using fitted OpenMax + model classifier on .npz embeddings."""
    device = torch.device(args.device)
    sep = args.sep

    # Load model
    model_save_dict = torch.load(args.model, map_location=device)
    model_config = model_save_dict["model_config"]
    model = ModelRegistry.get_model_class(args.head_model).from_config(model_config).to(device)
    model.load_state_dict(model_save_dict["model_state_dict"])
    model.eval()

    # Load OpenMax
    om = OpenMax.load(args.openmax_params)

    # Load embeddings + labels
    X = np.load(args.embeddings, allow_pickle=True)
    labels_df = pd.read_csv(args.labels, sep="\t", index_col=0)
    labels_df["Serotype"] = labels_df["Serotype"].fillna(DEFAULT_MISSING_LABEL)
    labels_df = labels_df[labels_df["Serotype"] != DEFAULT_MISSING_LABEL]

    keys = (
        labels_df["Is_capsule"].map(lambda x: "cbl" if x else "non-cbl")
        + sep
        + get_sample_id(labels_df)
    )
    valid_keys = [k for k in keys if k in X]
    valid_mask = keys.isin(valid_keys)
    labels_df = labels_df[valid_mask.values]
    keys = keys[valid_mask.values]

    X_filtered = np.stack([X[k] for k in keys])
    print(f"Loaded {len(X_filtered)} embeddings for prediction.")

    # Compute logits
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(X_filtered), args.batch_size):
            batch = torch.tensor(X_filtered[i : i + args.batch_size], dtype=torch.float32, device=device)
            sero_logits = model.serotype_classifier(batch)
            all_logits.append(sero_logits.cpu().numpy())
    all_logits = np.concatenate(all_logits, axis=0)

    # Score with OpenMax
    print("Running OpenMax scoring...")
    results = []
    for i in tqdm(range(len(X_filtered)), desc="OpenMax scoring"):
        score = om.openmax_score(X_filtered[i], all_logits[i])
        results.append({
            "sample_id": labels_df.index[i],
            "serotype": labels_df["Serotype"].iloc[i],
            "is_capsule": labels_df["Is_capsule"].iloc[i],
            "openmax_pred": score["openmax_pred"],
            "prob_unknown": round(score["prob_unknown"], 6),
            "is_novel": score["is_novel"],
            "raw_softmax_pred": score["raw_softmax_pred"],
        })

    results_df = pd.DataFrame(results)

    # Compute classification metrics if ground truth is available
    capsulated = results_df[results_df["is_capsule"].astype(bool)].copy()
    if len(capsulated) > 0:
        closed_correct = (capsulated["raw_softmax_pred"] == capsulated["serotype"]).mean()
        open_correct = (
            (capsulated["openmax_pred"] == capsulated["serotype"])
            | (capsulated["openmax_pred"] == OpenMax.UNKNOWN_LABEL)
        ).mean()
        novel_rate = capsulated["is_novel"].mean()
        print("\n--- OpenMax Summary (capsulated samples) ---")
        print(f"  Closed-set accuracy:  {closed_correct:.4f}")
        print(f"  Open-set accuracy:    {open_correct:.4f}")
        print(f"  Novel detection rate: {novel_rate:.4f}")
        print(f"  Mean P(unknown):      {capsulated['prob_unknown'].mean():.6f}")

    results_df.to_csv(args.output, index=False)
    print(f"\nPredictions saved to {args.output}")


# ──────────────────── Entrypoint ────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OpenMax open-set recognition for serotype classification"
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── fit ──
    p_fit = subparsers.add_parser("fit", help="Fit OpenMax parameters on training embeddings")
    p_fit.add_argument("--embeddings", required=True, help=".npz embeddings from infer_transformer")
    p_fit.add_argument("--labels", required=True, help="Final metadata TSV")
    p_fit.add_argument("--model", required=True, help="Trained model .pth")
    p_fit.add_argument("--output", required=True, help="Output .pkl for fitted OpenMax params")
    p_fit.add_argument("--device", default="cpu")
    p_fit.add_argument("--head_model", default=DEFAULT_HEAD_MODEL)
    p_fit.add_argument("--sep", default=DEFAULT_SEP)
    p_fit.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p_fit.add_argument("--alpha", type=int, default=10,
                        help="Number of top classes to revise (α in paper)")
    p_fit.add_argument("--tail_size", type=int, default=20,
                        help="Number of tail distances for Weibull fitting (η)")
    p_fit.add_argument("--distance_metric", default="euclid", choices=["euclid", "cosine"],
                        help="Distance metric for MAV comparisons")

    # ── predict ──
    p_pred = subparsers.add_parser("predict", help="Score samples using fitted OpenMax")
    p_pred.add_argument("--embeddings", required=True, help=".npz embeddings")
    p_pred.add_argument("--labels", required=True, help="Final metadata TSV")
    p_pred.add_argument("--model", required=True, help="Trained model .pth")
    p_pred.add_argument("--openmax_params", required=True, help="Fitted .pkl from fit mode")
    p_pred.add_argument("--output", required=True, help="Output CSV for predictions")
    p_pred.add_argument("--device", default="cpu")
    p_pred.add_argument("--head_model", default=DEFAULT_HEAD_MODEL)
    p_pred.add_argument("--sep", default=DEFAULT_SEP)
    p_pred.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)

    args = parser.parse_args()

    if args.command == "fit":
        cli_fit(args)
    elif args.command == "predict":
        cli_predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
