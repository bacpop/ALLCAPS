import json
import argparse
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

from .models import ModelRegistry
from .consts import DEFAULT_SEP, DEFAULT_BATCH_SIZE, DEFAULT_HEAD_MODEL

def main(args):
    sep = args.model_params.get("sep", DEFAULT_SEP)
    head_model = args.model_params.get("head_model", DEFAULT_HEAD_MODEL)

    print(f"Loading embeddings from: {args.embeddings}")
    X = np.load(args.embeddings, allow_pickle=True)  # shape: (N, L, D)
    labels = list(
        map(lambda sid: sid.split(sep), X.keys())
    )  # assuming keys are in the format "capsule_label|public_name"
    labels = pd.DataFrame(labels, columns=["Capsule_label", "Public_name"])
    labels["Capsule_label"] = (
        labels["Capsule_label"].map(lambda x: 1 if x == "cbl" else 0).astype(int)
    )  # TODO clean up this shit

    X = np.stack([X[k] for k in X.keys()])
    y = labels["Capsule_label"].values

    device = torch.device(args.device)
    print(f"Loading model from: {args.model}")
    model_save_dict = torch.load(args.model, map_location=device)
    model_config = model_save_dict['model_config']
    num_serotypes = model_save_dict['num_serotypes']
    
    print(f"Model configuration: {model_config}")
    print(f"Number of serotypes: {num_serotypes}")
    
    # Initialize model with saved configuration
    model = ModelRegistry.get_model_class(head_model) \
        .from_config(model_config) \
        .to(device)
    model.load_state_dict(model_save_dict['model_state_dict'])
    model.eval()

    print("Running inference...")
    all_logits, all_probs = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(X), args.batch_size)):
            batch = torch.tensor(
                X[i : i + args.batch_size], dtype=torch.float32, device=device
            )
            logits = model.cbl_classifier(batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_logits.append(preds)
            all_probs.append(probs)
    y_pred = np.concatenate(all_logits)
    y_score = np.concatenate(all_probs)

    clf_report = classification_report(
        y, y_pred, target_names=["Non-CBL", "CBL"], labels=[0, 1]
    )
    f1 = f1_score(y, y_pred, average="weighted")
    fpr, tpr, _ = roc_curve(y, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(args.output.replace(".txt", "_roc.pdf"))
    plt.close()

    print("Classification report:")
    print(clf_report)
    print("F1 score (weighted):", f1)
    with open(args.output, "w") as f:
        f.write("Classification report:\n")
        f.write(clf_report)
        f.write("\nF1 score (weighted): {}\n".format(f1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings", required=True, help="Path to .npy embeddings file (N, L, D)"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        default="cbl_classifier_output.txt",
        help="Output file for classification report",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--model_params", type=str, default="{}")

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

    main(args)
